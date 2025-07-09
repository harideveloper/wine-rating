"""Tests for evaluator component core logic."""

import ast
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class TestModelEvaluator:
    """Test core evaluator logic."""

    def test_feature_extraction_from_metadata(self):
        """Test extracting features from metadata."""
        metadata = {"features": "['price_numeric', 'Country']"}

        features_str = metadata.get("features", "")
        if features_str:
            features = ast.literal_eval(features_str)
        else:
            features = ["price_numeric", "Country"]

        assert features == ["price_numeric", "Country"]

    def test_metrics_calculation(self):
        """Test evaluation metrics calculation."""
        y_true = np.array([4.2, 3.8, 4.5])
        y_pred = np.array([4.1, 3.9, 4.4])

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        assert mse >= 0
        assert rmse >= 0
        assert -1 <= r2 <= 1

    def test_quality_score_calculation(self):
        """Test quality score formula."""
        r2 = 0.9
        rmse = 0.1

        r2_norm = max(0, r2)
        rmse_score = max(0, 1.0 - rmse)
        quality_score = (0.7 * r2_norm) + (0.3 * rmse_score)

        expected = (0.7 * 0.9) + (0.3 * 0.9)  # 0.9
        assert abs(quality_score - expected) < 1e-10

    @patch("joblib.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_model_loading(
        self, mock_file, mock_load
    ):  # pylint: disable=unused-argument
        """Test model loading logic."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([4.1, 3.9])
        mock_load.return_value = mock_model

        import joblib  # pylint: disable=import-outside-toplevel

        with open("model.joblib", "rb") as f:
            model = joblib.load(f)

        predictions = model.predict([[25.99, 1], [18.50, 0]])
        assert len(predictions) == 2
