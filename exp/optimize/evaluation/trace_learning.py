"""Sequential learning pass: the hybrid simulator learns trace by trace from fit traces.

The build's fit traces are replayed one at a time in their export (JSONL) order on the hybrid
simulator: learned memory supplies behavioral rules and the build's fit-only index supplies
recorded facts, never from the trace's own lineage. Each trace runs through the
existing learn-mode lifecycle of :func:`execute_fidelity_evaluation`: every step is predicted
from the task, the visible history and the current memory, the recorded reply is disclosed only
after the prediction commits, mismatches are investigated, the completed trajectory is
published as one memory version, and maintenance compresses the memory log. No second runner
exists. A rerun replays completed traces without provider calls and continues where it stopped.

Held-out lineages are reserved in the learning lineage before the first trace is registered, so
a held-out trace can never be learned. Because each fit trace is predicted before it is learned,
the saved reports also give a learning curve over consumption order at no extra cost.
"""

from collections.abc import Callable
from datetime import datetime

from pydantic import Field, JsonValue

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectBuildArtifacts, ProjectStore
from exp.common.tasks import TaskCase
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.fidelity import (
    FidelityMemory,
    FidelityRetrieval,
    FidelitySpec,
    PreparedFidelity,
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.fidelity_memory import DUPLICATE, INCOMPLETE, saved_memory_receipt
from exp.optimize.evaluation.trace_replay import (
    DEADLINE,
    ReplayCase,
    ReplayResult,
    attempts_used,
    content_run_id,
    recorded_exchanges,
    register_replay,
    replay_case,
    replay_result,
    replay_spec,
)
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy
from exp.simulation.world_model.experience_memory import memory_head, reserve_task_lineage
from exp.simulation.world_model.runtime import persist_fit_only_world_model
from exp.simulation.world_model.session_research import OnlineResearchPolicy

LABEL = "learn"
FEEDBACK_SOURCE = "recorded-trace-reply"
_PLAN_TYPE = "trace-learning-plan"


class LearningPlan(ContractModel):
    """The memory a project's build learns into, and how memory is recalled.

    Attributes:
        world_model: Memory-first simulator derived from the build's world model.
        lineage: Memory lineage, one per learning simulator.
        recall: Native JEV recall over memory records (not recorded-evidence retrieval).
    """

    world_model: ArtifactInput
    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    recall: FidelityRetrieval

    @property
    def selector_identity(self) -> str:
        """Identity of the memory selector this plan's recall constructs."""
        return JEVCueSelector(
            policy=self.recall.jev,
            top_k=self.recall.memory_top_k,
            decide=lambda request: {},
            record=lambda receipt: None,
        ).identity


class LearningOutcome(ContractModel):
    """What one learning pass did.

    Attributes:
        plan: Learning plan.
        traces: Fit traces in learning order.
        learned: Traces consumed into memory (published, or already consumed).
        incomplete: Traces whose every allowed attempt stopped before completing, so they
            were not consumed; the pass continues past them and reports them.
        head_version: Memory version after the pass; 0 before any publication.
        attempts_used: Physical attempts this invocation claimed.
        stopped: Why the pass stopped early, or None when every trace completed.
    """

    plan: LearningPlan
    traces: int
    learned: int
    incomplete: int = 0
    head_version: int
    attempts_used: int
    stopped: str | None = None


class CurveWindow(ContractModel):
    """Exact replies over one contiguous stretch of consumption order.

    Attributes:
        first: First trace position (1-based).
        last: Last trace position.
        exact: Exactly predicted replies.
        steps: Predicted steps.
    """

    first: int
    last: int
    exact: int
    steps: int

    @property
    def share(self) -> str:
        """Whole-number exact-reply percentage, or a dash without steps."""
        return f"{self.exact / self.steps:.0%}" if self.steps else "-"


class LearningCurve(ContractModel):
    """Accuracy of each fit trace's prediction, made before that trace was learned.

    Attributes:
        points: One result per learned trace, in consumption order.
        quarters: Four consumption windows; the boundary after the second is ~50% consumed.
        halves: First and second half of consumption.
    """

    points: tuple[ReplayResult, ...]
    quarters: tuple[CurveWindow, ...]
    halves: tuple[CurveWindow, ...]

    def line(self) -> str:
        """One line: exact replies per consumption quarter, then first versus second half."""
        quarters = " · ".join(f"{w.first}-{w.last} {w.share}" for w in self.quarters)
        line = f"Learning curve  exact replies by consumption order: {quarters}"
        if len(self.halves) == 2:
            line += f" (first half → second half {' → '.join(w.share for w in self.halves)})"
        return line


def learning_plan(project: ProjectStore, recall_alias: str | None = None) -> LearningPlan:
    """Derive the project's learning plan, persisting it on first use.

    Args:
        project: Project with a completed build.
        recall_alias: Native decision alias for memory recall; required the first time, and
            must match the recorded one afterwards. None reads the recorded plan.

    Returns:
        The plan for the build's current world model.

    Raises:
        ValueError: No completed build, no recorded plan and no alias, or a different alias.
    """
    build = _build(project)
    # Hybrid: learned memory supplies behavioral rules; the fit-only index supplies recorded
    # facts. Held-out traces are not in that index, and each task's own lineage is excluded
    # from retrieval, so a trace never retrieves its own recorded replies while it is learned.
    world_model = persist_fit_only_world_model(project.artifacts, build.world_model, build.fit_rag)
    lineage = "learn-" + sha256_json(world_model.model_dump(mode="json"))[:16]
    identity = stable_id(_PLAN_TYPE, {"lineage": lineage})
    if project.artifacts.exists(identity):
        saved = LearningPlan.model_validate_json(
            project.artifacts.read_bytes(identity, "plan.json")
        )
        if recall_alias is not None and saved.recall.jev_alias != recall_alias:
            raise ValueError(
                f"memory {lineage} learns with recall alias {saved.recall.jev_alias!r}; "
                "keep that alias, or rebuild to start a new memory"
            )
        return saved
    if recall_alias is None:
        raise ValueError(
            "no learned memory for this build; run exp build PROJECT --learn "
            "--memory-recall ALIAS first"
        )
    plan = LearningPlan(
        world_model=world_model,
        lineage=lineage,
        recall=FidelityRetrieval(jev_alias=recall_alias, jev=JEVPolicy()),
    )
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=_PLAN_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=project.artifacts.read(world_model.artifact_id).manifest.created_at,
            code_revision="trace-learning",
            inputs=(world_model,),
        ),
        files={"plan.json": plan.model_dump(mode="json")},
    )
    return plan


def fit_cases(project: ProjectStore) -> tuple[ReplayCase, ...]:
    """Every fit trace with something to replay, in the build's trace (JSONL) order.

    Each case's public task is the trace's own request: its task text, request context, public
    prefix, and published tools, or signatures observed in fit traces when it declares none.

    Args:
        project: Project with a completed build.

    Returns:
        Replay cases in learning order.
    """
    build = _build(project)
    bindings = {
        row.trace_id: row
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    traces = [
        trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
        if bindings[trace.trace_id].partition == "fit"
    ]
    calls = [
        (row.name, row.arguments)
        for trace in traces
        for row in recorded_exchanges(trace, bindings[trace.trace_id].lineage_id)
    ]
    cases = []
    for trace in traces:
        task = TaskCase(
            task_id=stable_id("learning-task", {"trace_id": trace.trace_id}),
            lineage_group_id=bindings[trace.trace_id].lineage_id,
            partition="fit",
            instruction=trace.task,
            initial_context=trace.initial_context,
            tools=trace.tools,
            workload_weight=1,
            source_trace_ids=(trace.trace_id,),
        )
        case = replay_case(task, trace, calls)
        if case is not None:
            cases.append(case)
    return tuple(cases)


def fit_calls(project: ProjectStore) -> list[tuple[str, JsonValue]]:
    """Recorded calls of every fit trace: the tool signatures a fit-trained simulator saw.

    Covers each fit trace's public prefix as well as its replayed steps, so this inventory
    matches the one the source-RAG profile derives from the fit-only index.

    Args:
        project: Project with a completed build.

    Returns:
        ``(tool name, arguments)`` pairs of fit traces.
    """
    calls: list[tuple[str, JsonValue]] = []
    for case in fit_cases(project):
        prefix = case.task.initial_context.get("public_prefix")
        for row in prefix if isinstance(prefix, list) else []:
            action = row.get("action") if isinstance(row, dict) else None
            if isinstance(action, dict) and isinstance(action.get("name"), str):
                calls.append((str(action["name"]), action.get("arguments")))
        calls.extend((row.name, row.arguments) for row in case.steps)
    return calls


def reserve_held_out(project: ProjectStore, plan: LearningPlan, *, code_revision: str) -> int:
    """Reserve every held-out lineage in the learning memory before anything is learned.

    Args:
        project: Project with a completed build.
        plan: Learning plan whose lineage is protected.
        code_revision: Recording implementation identity.

    Returns:
        Number of reserved held-out lineages.

    Raises:
        ValueError: A held-out lineage was already learned into this memory.
    """
    build = _build(project)
    held = sorted(
        {
            row.lineage_id
            for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
            if row.partition == "held_out"
        }
    )
    created = project.artifacts.read(plan.world_model.artifact_id).manifest.created_at
    for lineage in held:
        reserve_task_lineage(
            project.artifacts,
            plan.lineage,
            lineage,
            created_at=created,
            code_revision=code_revision,
        )
    return len(held)


def learning_spec(
    project: ProjectStore, plan: LearningPlan, case: ReplayCase, *, deadline: datetime
) -> FidelitySpec:
    """Learn-mode spec of one fit case, with a content-addressed run ID.

    Per-turn judging is off while learning: the judge still runs once per trace (the fidelity
    owner requires it), and the learning curve uses exact replies, not judge scores.

    Args:
        project: Project with a completed build.
        plan: Learning plan.
        case: Fit replay case.
        deadline: Absolute cutoff; excluded from the run identity.

    Returns:
        The spec.
    """
    return replay_spec(
        project,
        plan.world_model,
        case,
        deadline=deadline,
        label=LABEL,
        memory=FidelityMemory(lineage=plan.lineage, mode="learn"),
        retrieval=plan.recall,
        online_research=OnlineResearchPolicy(
            feedback_source=FEEDBACK_SOURCE, selection_policy_sha256=plan.selector_identity
        ),
        per_turn=False,
    )


def learn_from_fit_traces(
    project: ProjectStore,
    *,
    recall_alias: str,
    maximum_attempts: int,
    code_revision: str,
    clock: Callable[[], datetime],
    progress: ProgressHook | None = None,
    retry_incomplete: bool = False,
) -> LearningOutcome:
    """Learn from every fit trace in order, within an explicit physical-attempt budget.

    A trace is started only when its whole attempt ceiling fits the remaining budget, so the
    pass never stops inside a trace for lack of budget; rerunning continues from the first
    trace without a saved report. Completed traces replay without provider calls.

    Args:
        project: Project with a completed build.
        recall_alias: Native decision alias for memory recall.
        maximum_attempts: Physical attempts this invocation may claim, retries included.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for deadlines.
        progress: Shared progress display; reports ``Learning from traces k/n``.
        retry_incomplete: Replace an interrupted trace that already bought calls.

    Returns:
        The outcome; ``stopped`` explains an early stop.
    """
    plan = learning_plan(project, recall_alias)
    reserve_held_out(project, plan, code_revision=code_revision)
    cases = fit_cases(project)
    used, learned, incomplete, stopped = 0, 0, 0, None
    for index, case in enumerate(cases):
        report(progress, "Learning from traces", completed=index, total=len(cases))
        attempted: set[str] = set()
        while True:
            now = clock()
            spec = learning_spec(project, plan, case, deadline=now + DEADLINE)
            prepared, _ = register_replay(
                project,
                spec,
                label=LABEL,
                trace=case.trace_id,
                code_revision=code_revision,
                now=now,
                retry_incomplete=retry_incomplete,
            )
            if prepared.id in attempted:
                break  # attempts exhausted: the latest one stays the trace's result
            attempted.add(prepared.id)
            if saved_fidelity_report(project, prepared) is None:
                if used + spec.maximum_physical_attempts > maximum_attempts:
                    stopped = (
                        f"attempt budget reached after {learned} of {len(cases)} traces; "
                        "run again with a larger budget to continue"
                    )
                    break
                execute_fidelity_evaluation(project, prepared, clock=clock)
                used += attempts_used(project, prepared)
            else:
                # Replay completes any interrupted publication without provider calls.
                execute_fidelity_evaluation(project, prepared, clock=clock)
            receipt = saved_memory_receipt(project, prepared)
            if receipt is None or receipt.get("reason") != INCOMPLETE:
                break
        if stopped is not None:
            break
        receipt = saved_memory_receipt(project, prepared)
        if receipt is not None and receipt.get("reason") in ("consumed", DUPLICATE):
            learned += 1
        else:
            incomplete += 1
    if stopped is None:
        report(progress, "Learning from traces", completed=len(cases), total=len(cases))
    head = memory_head(project.artifacts, plan.lineage)
    return LearningOutcome(
        plan=plan,
        traces=len(cases),
        learned=learned,
        incomplete=incomplete,
        head_version=0 if head is None else head.version,
        attempts_used=used,
        stopped=stopped,
    )


def learning_curve(project: ProjectStore) -> LearningCurve | None:
    """Accuracy of fit-trace predictions in consumption order, from saved reports only.

    Args:
        project: Project with a recorded learning plan.

    Returns:
        The curve over learned traces, or None before anything was learned.
    """
    plan = learning_plan(project)
    points: list[ReplayResult] = []
    for case in fit_cases(project):
        found = _saved_learning_run(project, plan, case)
        if found is None:
            break
        points.append(found[1])
    if not points:
        return None
    return LearningCurve(
        points=tuple(points),
        quarters=_windows(points, 4),
        halves=_windows(points, 2) if len(points) > 1 else (),
    )


def _saved_learning_run(
    project: ProjectStore, plan: LearningPlan, case: ReplayCase
) -> tuple[PreparedFidelity, ReplayResult] | None:
    """The completed learning run of one case, found by content address without writing."""
    spec = learning_spec(project, plan, case, deadline=datetime.now().astimezone())
    generation = 1
    while True:
        run_id = content_run_id(spec, LABEL, case.trace_id, generation)
        plan_id = stable_id("fidelity", {"run_id": run_id})
        if not project.artifacts.exists(plan_id):
            return None
        prepared = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        saved = saved_fidelity_report(project, prepared)
        if saved is not None:
            return prepared, replay_result(project, case, prepared, saved)
        generation += 1


def _windows(points: list[ReplayResult], parts: int) -> tuple[CurveWindow, ...]:
    """Split consumption order into ``parts`` contiguous, nearly equal windows."""
    windows = []
    for part in range(parts):
        lo = part * len(points) // parts
        hi = (part + 1) * len(points) // parts
        if lo == hi:
            continue
        rows = points[lo:hi]
        windows.append(
            CurveWindow(
                first=lo + 1,
                last=hi,
                exact=sum(row.exact for row in rows),
                steps=sum(row.steps for row in rows),
            )
        )
    return tuple(windows)


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("learning needs a completed build; run exp build first")
    return build
