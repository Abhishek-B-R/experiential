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


def test_launch_tool_parser_matches_the_selected_completion_decoder() -> None:
    """Qwen XML and Hermes JSON actions require different native chat parsers."""
    qwen = VllmServerConfig(base=revision(), decoder="qwen35").command()
    hermes = VllmServerConfig(base=revision(), decoder="hermes").command()
    assert "--enable-auto-tool-choice" in qwen and "--enable-auto-tool-choice" in hermes
    assert qwen[qwen.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert qwen[qwen.index("--reasoning-parser") + 1] == "qwen3"
    assert hermes[hermes.index("--tool-call-parser") + 1] == "hermes"
    assert "--reasoning-parser" not in hermes
