"""Session-local investigations from explicitly disclosed post-prediction feedback."""

import copy
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import strict_equal
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore
from exp.runtime.sessions.store import SessionStore
from exp.simulation.engines.transition import (
    Completion,
    PreparedPrediction,
    ProposedTransition,
    TransitionEngine,
)
from exp.simulation.retrieval.selection import EvidencePolicy, SelectionQuery, selected_evidence

_INVESTIGATE = """Investigate one completed simulator prediction using only this input.
The disclosed observation is feedback about this past action, not a future answer or full
native state. Compare it with the prediction and candidate state. Describe the discrepancy
and propose a scoped, tentative hypothesis for later turns. Distinguish a wrong behavioral
rule from an incorrect current-state belief and from insufficient information. Do not invent
hidden values. Do not rewrite the world, change past scores, or promote a global rule.
You may supersede previous investigation IDs only when the supplied feedback contradicts
them. Preserve uncertainty when several explanations fit.
Fictional example: predicted {"status":"pending"}, observed {"status":"shipped"} for a
read of entity E. A supported note says the candidate's status belief about E was wrong at
that read; it does not establish that all entities are shipped or explain why E changed.
Return the required structured investigation, with supersedes:[] if nothing is disproved.
"""

_USE_RESEARCH = """
research_history.session contains post-prediction feedback and provisional investigations
from this session only. research_history.prior_trajectories contains provisional
investigations from earlier, separately simulated trajectories; their entity values belong
to those worlds, not this one. Observed replies concern the specified past action. Hypotheses
are not verified rules or complete native state. Use them to reconsider relevant assumptions
on later turns, preserving unrelated facts and chronology. Propose any resulting state changes
through the normal transition protocol. Never claim an automatic state correction occurred.
"""


class OnlineResearchPolicy(ContractModel):
    """Prospective limits and authority for supervised adaptation within one session.

    Attributes:
        feedback_source: Declared origin of disclosed past native replies.
        maximum_updates: Maximum model-generated investigations, not prediction retries.
        retained_notes: Maximum active investigations supplied to the investigator.
        selection_policy_sha256: Exact query-dependent memory selection policy.
        generate_hypotheses: False retains raw discrepancy feedback without an investigator call.
    """

    selection_policy_sha256: Sha256
    feedback_source: str = Field(min_length=1)
    generate_hypotheses: bool = Field(default=True, strict=True, exclude_if=bool)
    maximum_updates: int = Field(default=24, gt=0, strict=True)
    retained_notes: int = Field(default=24, gt=0, strict=True)


class Investigation(ContractModel):
    """Provisional interpretation that cannot directly edit a session or global knowledge.

    Attributes:
        discrepancy: Specific predicted-versus-observed difference.
        hypothesis: Tentative explanation or revised state belief.
        scope: Entities, tools and conditions to which the hypothesis might apply.
        uncertainty: Remaining alternatives or missing observations.
        supersedes: Earlier active note IDs contradicted by the supplied feedback.
    """

    discrepancy: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)
    supersedes: tuple[str, ...] = ()


class SessionResearch:
    """Retain append-only feedback and revisions without mutating candidate state.

    The same session lock serializes prediction and feedback. Immutable intents prevent
    an interrupted investigation from being silently purchased again after restart.
    Prior records come from a frozen memory version and are never edited by this session.
    """

    def __init__(
        self,
        *,
        session_id: str,
        policy: OnlineResearchPolicy | None,
        store: SessionStore,
        artifacts: ArtifactStore,
        complete: Completion | None,
        lock_path: Path,
        clock: Callable[[], datetime],
        code_revision: str,
        selection: EvidencePolicy,
        prior: Mapping[str, JsonObject] | None = None,
        prior_identity: JsonObject | None = None,
    ) -> None:
        """Bind feedback policy, metered investigator and an optional frozen prior memory.

        Args:
            session_id: Current trajectory's world identity.
            policy: Feedback learning policy; None only delivers the frozen prior memory.
            store: Shared durable session owner.
            artifacts: Immutable receipt owner.
            complete: Metered investigator, required exactly when a policy is present.
            lock_path: Session single-writer lease.
            clock: Aware clock.
            code_revision: Implementation identity.
            selection: Query-dependent selector over prior and session records.
            prior: Active research from earlier trajectories, keyed by record identity.
            prior_identity: Memory version bound into this session's identity.
        """
        if (policy is None) != (complete is None):
            raise ValueError("feedback learning requires both policy and investigator")
        if policy is not None and selection.identity != policy.selection_policy_sha256:
            raise ValueError("memory selector differs from registered online policy")
        if policy is None and not prior_identity:
            raise ValueError("research delivery without feedback requires a frozen memory")
        self._selection = selection
        self.session_id, self.policy = session_id, policy
        self._prior = copy.deepcopy(dict(prior or {}))
        self._prior_identity = copy.deepcopy(prior_identity)
        self._store, self._artifacts, self._complete = store, artifacts, complete
        self._lock = FileLock(lock_path, timeout=0)
        self._clock, self._code_revision = clock, code_revision

    @property
    def identity(self) -> Sha256:
        """Bind adaptation policy, prior memory and actual investigation and serving text."""
        return sha256_json(
            {
                "session": self.session_id,
                "policy": None if self.policy is None else self.policy.model_dump(mode="json"),
                "selection": self._selection.identity,
                "prior": self._prior_identity,
                "investigate": _INVESTIGATE,
                "use": _USE_RESEARCH,
                "schema": Investigation.model_json_schema(),
            }
        )

    def _record_id(self, operation_id: str) -> str:
        """Bind each observation to its session, immutable execution binding and operation."""
        binding = self._store.read_session(self.session_id).binding
        return stable_id(
            "session-research",
            {
                "session": self.session_id,
                "binding": binding.model_dump(mode="json"),
                "operation": operation_id,
                "policy": self.identity,
            },
        )

    def _records(self, transitions: list[JsonValue]) -> list[JsonObject]:
        """Load only feedback attached to already committed operations in this session."""
        records: list[JsonObject] = []
        if self.policy is None:
            return records
        for transition in transitions:
            if not isinstance(transition, dict) or not isinstance(
                transition.get("operation_id"), str
            ):
                raise ValueError("research requires committed operation identities")
            identity = self._record_id(str(transition["operation_id"]))
            if self._artifacts.exists(identity):
                records.append(json.loads(self._artifacts.read_bytes(identity, "record.json")))
            elif self._artifacts.exists(identity + "-intent"):
                raise ValueError("unfinished research update requires explicit recovery")
        return records

    def _active(self, records: list[JsonObject]) -> list[JsonObject]:
        """Render the current view while retaining superseded notes in immutable records."""
        active: dict[str, JsonObject] = {}
        for record in records:
            raw = record.get("investigation")
            if not isinstance(raw, dict):
                if (
                    self.policy is not None
                    and not self.policy.generate_hypotheses
                    and record.get("reply_matched") is False
                ):
                    active[str(record["id"])] = record
                continue
            update = Investigation.model_validate(raw)
            for identity in update.supersedes:
                active.pop(identity, None)
            active[str(record["id"])] = record
        return list(active.values())

    def observe(self, operation_id: str, observed_reply: JsonValue, source: str) -> JsonObject:
        """Learn from the latest completed single-tool reply, never from future actions.

        This API accepts explicitly disclosed reply feedback, not reference states or
        private implementations. The caller owns the source's authenticity. Replaying
        identical feedback returns the saved record without another model call.
        """
        policy, complete = self.policy, self._complete
        if policy is None or complete is None:
            raise ValueError("frozen memory sessions do not accept feedback")
        if source != policy.feedback_source:
            raise ValueError("feedback source differs from the registered disclosure policy")
        with self._lock:
            current = self._store.read_session(self.session_id)
            identity = self._record_id(operation_id)
            if self._artifacts.exists(identity):
                saved = json.loads(self._artifacts.read_bytes(identity, "record.json"))
                if not strict_equal(saved["observed_reply"], observed_reply):
                    raise ValueError("feedback replay changed the disclosed observation")
                return saved
            if self._artifacts.exists(identity + "-intent"):
                raise ValueError("unfinished research update requires explicit recovery")
            transitions = current.state.get("transitions")
            if not isinstance(transitions, list) or not transitions:
                raise ValueError("feedback requires a completed prediction")
            latest = transitions[-1]
            if not isinstance(latest, dict) or latest.get("operation_id") != operation_id:
                raise ValueError("new feedback must concern the latest completed prediction")
            replies = latest.get("replies")
            if not isinstance(replies, list) or len(replies) != 1:
                raise ValueError("reply feedback requires a single-tool prediction")
            if self._clock() >= current.binding.deadline:
                raise TimeoutError("session deadline reached before investigation")
            records = self._records(transitions)
            matched = strict_equal(replies[0], observed_reply)
            record: JsonObject = {
                "id": identity,
                "session_id": self.session_id,
                "operation_id": operation_id,
                "actions": latest["actions"],
                "committed_version": current.version,
                "source": source,
                "transition_sha256": sha256_json(latest),
                "predicted_reply": replies[0],
                "observed_reply": observed_reply,
                "reply_matched": matched,
                "authority": "disclosed-past-reply; hypotheses-unverified",
                "investigation": None,
            }
            if not matched and policy.generate_hypotheses:
                if sum(isinstance(row.get("investigation"), dict) for row in records) >= (
                    policy.maximum_updates
                ):
                    raise ValueError("session investigation budget exhausted")
                active = self._active(records)[-policy.retained_notes :]
                body: JsonObject = {
                    "messages": [
                        {"role": "system", "content": _INVESTIGATE},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "feedback": record,
                                    "candidate_transition": latest,
                                    "previous_investigations": active,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        },
                    ],
                    "response_schema": Investigation.model_json_schema(),
                }
                self._save(identity + "-intent", {"request": body})
                proposal = complete(body, identity + "-investigate")
                self._save(identity + "-proposal", {"proposal": proposal})
                update = Investigation.model_validate(proposal)
                available = {str(row["id"]) for row in active}
                if len(set(update.supersedes)) != len(update.supersedes) or not set(
                    update.supersedes
                ).issubset(available):
                    raise ValueError("investigation can supersede only supplied active notes")
                if self._clock() >= current.binding.deadline:
                    raise TimeoutError("late investigation retained but not activated")
                record["investigation"] = update.model_dump(mode="json")
            self._save(identity, record)
            return copy.deepcopy(record)

    def deliver(
        self, history: tuple[JsonObject, ...], action: ToolCall, candidate_state: JsonObject
    ) -> tuple[list[JsonObject], JsonObject]:
        """Select among prior-trajectory and active session notes for this exact chronology."""
        state = self._store.read_session(self.session_id).state
        if not strict_equal(state.get("history"), list(history)):
            raise ValueError("research delivery history differs from the committed session")
        transitions = state.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("research requires a committed transition inventory")
        if self.policy is not None and (
            self._selection.identity != self.policy.selection_policy_sha256
        ):
            raise ValueError("online memory selection policy changed")
        active = self._active(self._records(transitions))
        catalog: dict[str, JsonValue] = dict(self._prior)
        catalog.update({str(row["id"]): row for row in active})
        selected, receipt = selected_evidence(
            self._selection,
            SelectionQuery(action=action, state=candidate_state, history=history),
            catalog,
        )
        return [value for value in selected.values() if isinstance(value, dict)], receipt

    def _save(self, identity: str, payload: JsonObject) -> None:
        """Publish immutable receipts outside business-state storage."""
        self._artifacts.write_json(
            artifact_id=identity,
            artifact_type="session-research",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=self._clock(), code_revision=self._code_revision
            ),
            files={"record.json": payload},
        )


class ResearchTransitionEngine:
    """Deliver session investigations through the existing grounded request and parser."""

    def __init__(self, base: TransitionEngine, research: SessionResearch) -> None:
        """Keep prediction semantics and output parsing with the existing engine."""
        self._base, self._research = base, research

    @property
    def identity(self) -> Sha256:
        """Freeze the policy while its session-local feedback grows."""
        return sha256_json({"engine": self._base.identity, "research": self._research.identity})

    def prepare(
        self, state: JsonObject, history: tuple[JsonObject, ...], actions: tuple[ToolCall, ...]
    ) -> PreparedPrediction:
        """Add only prior investigations and bind the exact delivered memory to receipts."""
        prepared = self._base.prepare(state, history, actions)
        body = copy.deepcopy(prepared.body)
        messages = body.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("session research requires the grounded two-message protocol")
        system, user = messages
        if not isinstance(system, dict) or not isinstance(user, dict):
            raise ValueError("invalid grounded message inventory")
        payload = json.loads(str(user["content"]))
        if len(actions) != 1:
            raise ValueError("online research supports one tool action per prediction")
        notes, selection = self._research.deliver(history, actions[0], state)
        context = payload.setdefault("research_history", {})
        if not isinstance(context, dict):
            raise ValueError("research context must be an object")
        context["session"] = [n for n in notes if n.get("session_id") == self._research.session_id]
        context["prior_trajectories"] = [
            n for n in notes if n.get("session_id") != self._research.session_id
        ]
        system["content"] = str(system["content"]) + _USE_RESEARCH
        user["content"] = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return PreparedPrediction(
            body=body,
            evidence={
                **prepared.evidence,
                "request_sha256": sha256_json(body),
                "session_research_sha256": sha256_json(notes),
                "session_research_ids": [row["id"] for row in notes],
                "feedback_policy_sha256": self._research.identity,
                "session_research_selection": selection,
            },
        )

    def parse(
        self,
        proposal: JsonObject,
        actions: tuple[ToolCall, ...],
        *,
        current_state: JsonObject | None = None,
    ) -> ProposedTransition:
        """Preserve the original proposal validation and state-effect semantics."""
        return self._base.parse(proposal, actions, current_state=current_state)
