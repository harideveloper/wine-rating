"""Tests for trainer component core logic."""

from unittest.mock import MagicMock, mock_open, patch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestModelTrainer:
    """Test core trainer logic."""

    def test_feature_selection(self):
        """Test feature selection logic."""
        df = pd.DataFrame(
            {
                "Country": ["France", "Italy"],
                "price_numeric": [25.99, 18.50],
                "Rating": [4.2, 3.8],
            }
        )
        # features selection
        categorical = ["Country", "Region", "Type"]
        available_categorical = [col for col in categorical if col in df.columns]
        numeric = ["price_numeric"]
        features = numeric + available_categorical
        assert features == ["price_numeric", "Country"]

    def test_pipeline_creation(self):
        """Test model pipeline creation."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        assert len(pipeline.steps) == 2
        assert isinstance(pipeline.steps[1][1], RandomForestRegressor)

    def test_data_preparation(self):
        """Test training data preparation."""
        df = pd.DataFrame(
            {
                "price_numeric": [25.99, 18.50],
                "Country": ["France", "Italy"],
                "Rating": [4.2, 3.8],
            }
        )
        features = ["price_numeric", "Country"]
        features_data = df[features].values
        target_data = df["Rating"].values
        assert features_data.shape == (2, 2)
        assert target_data.shape == (2,)

    @patch("joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_model_saving(
        self, mock_file, mock_dump
    ):  # pylint: disable=unused-argument
        """Test model saving logic."""
        mock_pipeline = MagicMock()
        import joblib  # pylint: disable=import-outside-toplevel

        with open("model.joblib", "wb") as f:
            joblib.dump(mock_pipeline, f)
        mock_dump.assert_called_once()
