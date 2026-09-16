"""Launch configuration binds identity and removes probability-changing defaults."""

from exp.runtime.claas.serving.configuration import VllmServerConfig
from exp.runtime.claas.serving.vllm_test import revision


def test_exact_server_configuration_pins_both_revisions() -> None:
    """No implicit tokenizer, generation config, or public control listener is selected."""
    command = VllmServerConfig(base=revision()).command()
    assert command[command.index("--revision") + 1] == "a" * 40
    assert command[command.index("--tokenizer-revision") + 1] == "b" * 40
    assert command[command.index("--generation-config") + 1] == "vllm"
    assert command[command.index("--host") + 1] == "127.0.0.1"
