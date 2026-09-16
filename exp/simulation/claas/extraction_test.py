"""Protocol normalization without inferred tool-error labels."""

from exp.simulation.claas.extraction import tool_results
from exp.simulation.claas.mining_test import make_experience


def test_error_words_in_tool_content_do_not_become_structured_failure_labels() -> None:
    """Arbitrary prose may discuss errors without reporting an execution failure."""
    source = make_experience(result_call="a", result_content="There are no errors in this report.")
    assert tool_results(source)[0].is_error is None
