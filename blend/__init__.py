"""Blend - 极致成本效率商用 API.

Minimax 极致压缩 + Claude 质量保障 + 固定成本.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so all providers see API keys
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

from blend.core.orchestrator import BlendOrchestrator, OrchestratorResult

__version__ = "2.1.0"
__author__ = "Blend Team"
__description__ = "极致成本效率商用 API"

__all__ = ["BlendOrchestrator", "OrchestratorResult"]
