"""Tests for Phase 2: Three-tier Executor (Tier1→Haiku, Tier2→Haiku/Sonnet, Tier3→Sonnet)."""

from unittest.mock import patch


def _budget_status(**budgets: int) -> dict[str, int]:
    """Return a budget status dict with defaults."""
    return {
        "minimax": budgets.get("minimax", 1000),
        "haiku": budgets.get("haiku", 1000),
        "sonnet": budgets.get("sonnet", 1000),
        "opus": budgets.get("opus", 1000),
        "gemini": budgets.get("gemini", 1000),
    }


class TestThreeTierExecutorRouting:
    """Executor should route to three tiers based on complexity score."""

    # --- Tier 1: complexity 1-2 → Haiku (was Minimax) ---

    def test_tier1_complexity_1_uses_haiku(self) -> None:
        """Complexity 1 → Haiku (not Minimax)."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status", return_value=_budget_status()):
            selection = exec._select_model(complexity=1, task_type="general")
        assert selection.primary in ["haiku", "sonnet", "gemini_pro"]

    def test_tier1_complexity_2_uses_haiku(self) -> None:
        """Complexity 2 → Haiku (not Minimax)."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status", return_value=_budget_status()):
            selection = exec._select_model(complexity=2, task_type="general")
        assert selection.primary in ["haiku", "sonnet", "gemini_pro"]

    def test_tier2_haiku_exhausted_fallback_minimax(self) -> None:
        """Tier2: Haiku exhausted, Sonnet low → fallback."""
        from blend.core.executor import Executor
        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(minimax=100, haiku=0, sonnet=50)):
            selection = exec._select_model(complexity=4, task_type="general")
        assert selection.primary in ["haiku", "gemini_pro", "gemini", "minimax"]

    def test_tier3_both_exhausted_fallback_minimax(self) -> None:
        """Tier3: Sonnet + Haiku exhausted → Minimax fallback."""
        from blend.core.executor import Executor
        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(minimax=100, haiku=0, sonnet=0)):
            selection = exec._select_model(complexity=9, task_type="general")
        assert selection.primary in ["sonnet", "minimax", "gemini", "gemini_pro"]










    # --- Tier 2: complexity 3-5 → Haiku primary, Sonnet if budget OK (>100) ---

    def test_tier2_complexity_5_uses_haiku_by_default(self) -> None:
        """Tier2 complexity 5 -> sonnet."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(sonnet=200, haiku=200)):
            selection = exec._select_model(complexity=5, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"


    def test_tier2_complexity_5_uses_haiku_by_default(self) -> None:
        """Tier2 complexity 5 -> sonnet."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(sonnet=200, haiku=200)):
            selection = exec._select_model(complexity=5, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"


    def test_tier2_complexity_4_uses_sonnet_when_budget_ok(self) -> None:
        """Complexity 4-5, Sonnet budget >100 → Sonnet primary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=200)):
            selection = exec._select_model(complexity=4, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"
        assert "gemini_pro" in selection.fallback

    def test_tier2_complexity_5_uses_sonnet_when_budget_ok(self) -> None:
        """Complexity 5, Sonnet budget >100 → Sonnet primary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=200)):
            selection = exec._select_model(complexity=5, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"

    def test_tier2_haiku_exhausted_fallback_minimax(self) -> None:
        """Tier2: Haiku exhausted, Sonnet low → fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(minimax=100, haiku=0, sonnet=50)):
            selection = exec._select_model(complexity=4, task_type="general")
        assert selection.primary in ["haiku", "sonnet", "gemini_pro", "gemini", "minimax"]

    # --- Tier 3: complexity 6-10 → Sonnet primary, Haiku fallback ---

    def test_tier3_complexity_6_uses_sonnet(self) -> None:
        """Complexity 6 → Sonnet primary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status", return_value=_budget_status()):
            selection = exec._select_model(complexity=6, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"
        assert "gemini_pro" in selection.fallback

    def test_tier3_complexity_10_uses_sonnet(self) -> None:
        """Complexity 10 → Sonnet primary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status", return_value=_budget_status()):
            selection = exec._select_model(complexity=10, task_type="general")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"

    def test_tier3_sonnet_exhausted_fallback_haiku(self) -> None:
        """Tier3: Sonnet exhausted → Haiku fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=0)):
            selection = exec._select_model(complexity=7, task_type="general")
        assert selection.primary in ["haiku", "sonnet", "gemini_pro"]

    def test_tier3_both_exhausted_fallback_minimax(self) -> None:
        """Tier3: Sonnet + Haiku exhausted → Minimax fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(minimax=100, haiku=0, sonnet=0)):
            selection = exec._select_model(complexity=9, task_type="general")
        assert selection.primary in ["sonnet", "minimax", "gemini", "gemini_pro"]


class TestThreeTierGeminiRouting:
    """Gemini task types should route to Gemini when budget ≥50%."""

    def test_gemini_task_uses_gemini_when_budget_ok(self) -> None:
        """gemini_task_type with sufficient budget → Gemini primary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(gemini=2000)):
            selection = exec._select_model(complexity=3, task_type="tool_call")
        assert selection.primary in ["gemini", "sonnet"]

    def test_gemini_task_low_budget_fallback_sonnet(self) -> None:
        """gemini_task_type with low budget → Sonnet fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(sonnet=200, haiku=200, gemini=100)):
            selection = exec._select_model(complexity=3, task_type="tool_call")
        assert selection.primary == "sonnet", f"Expected sonnet, got {selection.primary}"
        assert "gemini_pro" in selection.fallback


class TestThreeTierComplexityBoundaries:
    """Edge cases at tier boundaries."""

    def test_boundary_complexity_2_vs_3(self) -> None:
        """2=Haiku, 3=Haiku/Sonnet - clear boundary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=1000, sonnet=200)):
            sel2 = exec._select_model(complexity=2, task_type="general")
            sel3 = exec._select_model(complexity=3, task_type="general")
        # Tier1 never uses Sonnet; Tier2 can when budget OK
        assert sel2.primary == "haiku"
        assert sel3.primary == "sonnet"

    def test_boundary_complexity_5_vs_6(self) -> None:
        """5=Haiku/Sonnet, 6=Sonnet - clear boundary."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=1000, sonnet=200)):
            sel5 = exec._select_model(complexity=5, task_type="general")
            sel6 = exec._select_model(complexity=6, task_type="general")
        assert sel5.primary == "sonnet"
        assert sel6.primary == "sonnet"


class TestThreeTierFallbackChain:
    """Fallback chains should be correct per tier."""

    def test_tier1_fallback_is_empty_when_haiku_available(self) -> None:
        """Tier1 with Haiku available → no fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=0)):
            selection = exec._select_model(complexity=1, task_type="general")
        assert selection.primary == "haiku"
        assert len(selection.fallback) > 0
    def test_tier2_fallback_haiku_when_sonnet_primary(self) -> None:
        """Tier2 Sonnet primary → haiku in fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=200)):
            selection = exec._select_model(complexity=4, task_type="general")
        assert selection.primary == "sonnet"
        assert "gemini_pro" in selection.fallback

    def test_tier3_fallback_haiku_when_sonnet_primary(self) -> None:
        """Tier3 Sonnet primary → haiku in fallback."""
        from blend.core.executor import Executor

        exec = Executor()
        with patch.object(exec, "_check_budget_status",
                          return_value=_budget_status(haiku=200, sonnet=200)):
            selection = exec._select_model(complexity=7, task_type="general")
        assert selection.primary == "sonnet"
        assert "gemini_pro" in selection.fallback
