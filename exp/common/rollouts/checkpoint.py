"""Provider-neutral state for continuation at completed text-world turn boundaries."""

from exp.common.core.artifacts import ContractModel
from exp.common.models import ModelMessage, ModelResponse, OperationEconomics


class TextRolloutCheckpoint(ContractModel):
    """Exact visible history and paid responses, excluding customer-agent process state.

    Only the built-in stateless chat runtime may restore this checkpoint. A checkpoint is
    omitted when redaction would change state or a candidate/world turn is unfinished.

    Attributes:
        visible_transcript: Ordered assistant and environment messages at the turn boundary.
        candidate_responses: Already-paid worker responses, retained without redispatch.
        world_model_responses: Paid environment responses, including private state transitions.
        retrieval_economics: Metered retrieval operations incurred by this prefix.
    """

    visible_transcript: tuple[ModelMessage, ...]
    candidate_responses: tuple[ModelResponse, ...]
    world_model_responses: tuple[ModelResponse, ...]
    retrieval_economics: tuple[OperationEconomics, ...]
