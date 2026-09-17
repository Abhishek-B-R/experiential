"""Add nullable cache-write evidence without rewriting past accounting."""

CACHE_WRITE_COLUMNS = (
    "cache_creation_input_rate",
    "cache_creation_1h_input_rate",
    "long_context_cache_creation_input_rate",
    "long_context_cache_creation_1h_input_rate",
    "preferred_cache_creation_input_rate",
    "preferred_cache_creation_1h_input_rate",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
)
"""Frozen nano-USD rates and observed token counts; existing rows remain NULL."""

CACHE_WRITE_MIGRATION = tuple(
    f"ALTER TABLE gateway_attempts ADD COLUMN {column} INTEGER "
    f"CHECK ({column} IS NULL OR {column} >= 0)"
    for column in CACHE_WRITE_COLUMNS
)
