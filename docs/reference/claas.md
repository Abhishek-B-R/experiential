# CLaaS continual learning

CLaaS trains a scoped LoRA adapter from supplied scenarios, environments, and feedback. It preserves
the exact student rollout, delegates optimization to public veRL workers, compares the candidate and
active policies on frozen held-out tasks, and activates a candidate only when the configured
promotion checks pass. Serving pauses and drains before an update, then resumes with the selected
revision. Each user/application pair owns its adapter state and rollback history.

| Component | Responsibility |
|---|---|
| Rust gateway | Bounded traffic capture, feedback, episode membership, scoped admission, and serving identity. |
| CLaaS core | Exact-rollout validation, bounded training, paired evaluation, adapter state, activation, and rollback. |
| Application environment | Scenario construction, tool execution or simulation, termination, and reward assignment. |
| Public veRL worker | Backward passes, optimizer updates, scheduling, and resumable training checkpoint state. |
| Modal adapter | Remote compute lifecycle and verified input/checkpoint transport. |

## Process lifetime and update pauses

Start the private vLLM server once with the generated launch arguments and keep it running across
cycles. The serving controller unloads the scoped LoRA and uses level-1 sleep before training, then
wakes the same server and loads the selected revision. This avoids restarting the inference process;
it still incurs weight-transfer, cache-refill, and adapter-loading costs.

The local training adapter starts a fresh veRL worker subprocess for each batch, restores native
actor/teacher/optimizer state, performs the bounded update, saves state, and exits. The Modal adapter
runs that same worker in a single-use container. Neither adapter currently keeps a resident training
engine. Public inference stays paused throughout practice, training, and paired evaluation; feedback
storage remains available. Asynchronous APIs do not imply concurrent public serving and training.

Persistent training sessions and separate-GPU concurrent execution are future runtime extensions.
They can use the same environment, training, serving, admission, and checkpoint contracts, but need
explicit resource coordination and GPU validation before either behavior is claimed.

## Supplying an environment

Define tasks with `exp.common.claas.scenarios.Scenario`: scope, stable scenario and environment IDs,
policy-visible messages/tools, and private `environment_data`. Only messages and tools reach the
policy. Source-traffic IDs are optional, so authored or imported tasks need no capture store or
generation step.

Implement `exp.runtime.environments.learning.Environment.open(scenario)` to reset a session. Its
`step(action)` returns an `EnvironmentTransition` with visible messages, a terminal flag, and
optional private scalar/text feedback. `close(reason)` releases resources and returns retained
evidence. A distinct session runs for each practice or evaluation attempt.

This asynchronous learning contract consumes complete assistant actions and returns termination
and private feedback. The existing `exp.runtime.environments.interface.EnvironmentRuntime` is a
synchronous tool executor: its context-managed sessions accept individual `ToolCall` values and
return observations. An application can wrap that executor in a learning environment by supplying
episode termination, reward assignment, and asynchronous cleanup.

An `EpisodeScorer` scores and verifies retained episodes. `EnvironmentEvaluator` adapts it to paired
evaluation; an application may instead implement `TaskEvaluator` directly. Freeze the held-out
cohort before training and keep it separate from fit scenarios. The environment owns reward
assignment, and the evaluation scorer checks the domain outcome. Neither requires an LLM judge.

Given fit/held-out scenarios, an environment, a scorer, initialized serving/admission controllers,
and a selected training backend factory, the learning call is:

```python
from exp.optimize.claas.lifecycle.cycle import run_cycle
from exp.optimize.claas.lifecycle.inputs import PreparedCycle
from exp.optimize.claas.evaluation.environment import EnvironmentEvaluator

evaluator = EnvironmentEvaluator(environment, scorer, maximum_steps=8)
plan = PreparedCycle(
    scenarios=fit_scenarios,
    environment=environment,
    evaluation=evaluator.freeze(held_out_scenarios),
    evaluator=evaluator,
    external_reservation_usd=external_reservation_usd,
)
result = await run_cycle(
    directory=application_directory,
    config=learning_config,
    plan=plan,
    serving=serving,
    admission=admission,
    backend_factory=backend_factory,
    compute_reservation_usd=compute_reservation_usd,
)
```

The application supplies conservative reservations for environment/evaluator calls and training
compute. Zero is appropriate only for operations that incur no charge. A `CycleSource` can instead
declare the full external reservation before its `prepare` method assembles a `PreparedCycle`
under the core's application lock. This keeps source selection and held-out reservations consistent.
The source's bounded context is persisted and hash-bound to the cycle report before training.

## Local setup and optional traffic workflow

`exp optimize claas init APPLICATION` configures pinned base/tokenizer revisions, LoRA settings,
finite cycle limits, and the compute selection. Its generic `config.json` requires no provider
alias. Supplying both `--world-model` and `--judge` additionally selects the traffic workflow and
writes `traffic-workflow.json`. That workflow composes source selection, failure mining, scenario
synthesis, a world-model harness, and provider-backed evaluation outside the learning core.

The CLI commands are `init`, `status`, `capture`, `bind`, `activate`, `train`, and `rollback`.
`capture` opts an application's authenticated gateway traffic into bounded storage. `bind` writes
private runtime settings and shows the student vLLM launch arguments; `activate` verifies serving
readiness before opening admission. The CLI `train` command runs the configured traffic workflow.
Other environments use the Python API above. `rollback` drains admission and restores the preceding
verified revision.

The JSON passed to `bind --execution` nests traffic provider ceilings under `traffic`; these are
stored in `traffic-providers.json`, separately from student/worker `execution.json`. Local worker
settings are checked before provider credentials. Each cycle freezes its current evaluation cohort
and estimates every scheduled provider call before constructing providers. The configured command
ceiling is hard; `--yes` cannot exceed it. Selecting the workflow's providers authorizes the
corresponding source/evaluation disclosure.

Feedback can remain absent or contain binary, scalar, or text signals. Captured response IDs and
explicit episode membership determine ownership; unrelated requests are never joined by transcript
prefix. Private scalar and text feedback stay outside policy-visible environment messages.

## Evaluation isolation and validation

Applications that send external held-out evaluation through the captured gateway can use
`exp.optimize.workflows.traffic_learning.sources.holdouts` while holding the application cycle lock.
`begin_evaluation_holdout` persists a pending marker before dispatch;
`acknowledge_evaluation_holdout` atomically reserves the returned response ID and clears that exact
marker. `reserve_evaluation_holdout` reserves already known IDs. Traffic preparation excludes every
linked group touching a reserved ID before splitting or synthesis. An unresolved request blocks
learning after restart; no timeout clears it. Preserve `evaluation-holdouts.json` and the capture
database for reconciliation instead of deleting a pending marker. Previously fitted responses
cannot be relabeled as held out.

The optional `claas-verl` extra supplies public veRL `TrainingWorker` and FSDP execution. CLaaS adds
its SDPO/REINFORCE/hybrid objective and feedback-teacher extension at that worker boundary. Modal
runs the same worker. Neither compute selection changes the algorithm or chooses an environment.

Deterministic tests cover environment lifecycle, exact-token validation, objective calculations,
worker/checkpoint contracts, isolation, failure recovery, and promotion decisions. Fixture backend
receipts prove orchestration only. Actual CUDA training, full-size serving, paid provider calls,
Modal GPU execution, and measured learning improvement remain unrun for this feature.
