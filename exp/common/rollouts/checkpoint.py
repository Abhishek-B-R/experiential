"""Provider-neutral state for continuation at completed text-world turn boundaries."""

from exp.common.core.artifacts import ContractModel
from exp.common.models import ModelMessage, ModelResponse, OperationEconomics


class TextRolloutCheckpoint(ContractModel):
    """Exact visible history and paid responses, excluding customer-agent process state.

    Only the built-in stateless chat runtime may restore this checkpoint. A checkpoint is
    omitted when redaction would change state or a candidate/world turn is unfinished.
    """

    visible_transcript: tuple[ModelMessage, ...]
    candidate_responses: tuple[ModelResponse, ...]
    world_model_responses: tuple[ModelResponse, ...]
    retrieval_economics: tuple[OperationEconomics, ...]
