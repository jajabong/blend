"""Prompt templates for L2 strategy injection."""

# System prompt for L2 strategy injection into L3 execution.
# Used in executor._call_model when strategy.plan is provided.
L2_STRATEGY_SYSTEM_TEMPLATE = """请按以下计划执行:
{plan_text}"""

# Comma-separated list of supported placeholders
L2_STRATEGY_PLACEHOLDERS = ("plan_text",)
