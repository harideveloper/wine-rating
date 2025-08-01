"""Tests for promotion gate component."""

from unittest.mock import MagicMock
from pipelines.components.model_promotion_gate import promotion_gate

promotion_gate_func = promotion_gate.python_func


def test_promotion_gate_passes_all_criteria():
    """Test promotion gate passes when all criteria are met."""
    # Mock with valid metadata
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.999",  # Above threshold
        "ready_for_promotion": "true",
    }

    # Test with default threshold (0.99)
    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.99
    )

    assert result is True


def test_promotion_gate_passes_exact_threshold():
    """Test promotion gate passes when quality score equals threshold."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.99",  # Exactly at threshold
        "ready_for_promotion": "true",
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.99
    )

    assert result is True


def test_promotion_gate_fails_eval_status_not_completed():
    """Test promotion gate fails when evaluation status is not completed."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "failed",  # Not completed
        "quality_score": "0.96",
        "ready_for_promotion": "true",
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    assert result is False


def test_promotion_gate_fails_quality_score_below_threshold():
    """Test promotion gate fails when quality score is below threshold."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.94",  # Below threshold
        "ready_for_promotion": "true",
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.99
    )

    assert result is False


def test_promotion_gate_fails_not_ready_for_promotion():
    """Test promotion gate fails when ready_for_promotion is false."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.96",
        "ready_for_promotion": "false",  # Not ready
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    assert result is False


def test_promotion_gate_handles_case_insensitive_ready_flag():
    """Test promotion gate handles different cases for ready_for_promotion."""
    test_cases = ["true", "TRUE", "True", "TrUe"]

    for ready_value in test_cases:
        mock_fetched_model = MagicMock()
        mock_fetched_model.metadata = {
            "display_name": "wine-quality-model",
            "eval_status": "completed",
            "quality_score": "0.96",
            "ready_for_promotion": ready_value,
        }

        result = promotion_gate_func(
            fetched_model=mock_fetched_model, promotion_threshold=0.95
        )

        assert result is True, f"Should pass with ready_for_promotion='{ready_value}'"


def test_promotion_gate_fails_multiple_criteria():
    """Test promotion gate fails when multiple criteria are not met."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "running",  # Not completed
        "quality_score": "0.85",  # Below threshold
        "ready_for_promotion": "false",  # Not ready
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    assert result is False


def test_promotion_gate_handles_missing_metadata_gracefully():
    """Test promotion gate handles missing metadata with defaults."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model"
        # Missing eval_status, quality_score, ready_for_promotion
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    # Should fail with defaults (empty string != "completed", 0.0 < 0.95, "false" != "true")
    assert result is False


def test_promotion_gate_handles_invalid_quality_score():
    """Test promotion gate handles invalid quality score format."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "invalid_score",  # Invalid format
        "ready_for_promotion": "true",
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    # Should fail due to invalid quality score
    assert result is False


def test_promotion_gate_with_custom_threshold():
    """Test promotion gate with custom threshold values."""
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.85",
        "ready_for_promotion": "true",
    }

    # Test with lower threshold - should pass
    result_low = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.80
    )
    assert result_low is True

    # Test with higher threshold - should fail
    result_high = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.90
    )
    assert result_high is False


def test_promotion_gate_handles_exception():
    """Test promotion gate handles exceptions gracefully."""
    # Mock that raises exception when accessing metadata
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = None  # Will cause exception

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )

    # Should return False on exception
    assert result is False


def test_promotion_gate_edge_cases():
    """Test promotion gate with various edge cases."""

    # Test with zero quality score
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "eval_status": "completed",
        "quality_score": "0.0",  # Zero score
        "ready_for_promotion": "true",
    }

    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.95
    )
    assert result is False

    # Test with perfect quality score
    mock_fetched_model.metadata["quality_score"] = "1.0"
    result = promotion_gate_func(
        fetched_model=mock_fetched_model, promotion_threshold=0.99
    )
    assert result is True


def test_promotion_gate_eval_status_variations():
    """Test promotion gate with different eval_status values."""
    eval_statuses = ["running", "pending", "failed", "error", "unknown", ""]

    for status in eval_statuses:
        mock_fetched_model = MagicMock()
        mock_fetched_model.metadata = {
            "display_name": "wine-quality-model",
            "eval_status": status,
            "quality_score": "0.96",
            "ready_for_promotion": "true",
        }

        result = promotion_gate_func(
            fetched_model=mock_fetched_model, promotion_threshold=0.95
        )

        # Only "completed" should pass
        assert result is False, f"Should fail with eval_status='{status}'"
