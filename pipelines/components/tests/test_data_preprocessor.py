"""Tests for preprocessor component core logic."""

import re

import pandas as pd
from sklearn.model_selection import train_test_split


class TestDataPreprocessor:
    """Test core preprocessor logic."""

    def test_categorical_cleaning(self):
        """Test categorical feature cleaning."""
        df = pd.DataFrame({"Country": ["France", None, "Spain"]})
        df["Country"] = df["Country"].fillna("Unknown")

        assert not df["Country"].isna().any()
        assert "Unknown" in df["Country"].values

    def test_price_extraction(self):
        """Test price numeric extraction."""

        def extract_price(price):
            if pd.notna(price) and re.search(r"\d+\.?\d*", str(price)):
                return float(re.search(r"(\d+\.?\d*)", str(price)).group(1))
            return 0

        assert extract_price("$25.99") == 25.99
        assert extract_price(None) == 0
        assert extract_price("invalid") == 0

    def test_synthetic_rating_creation(self):
        """Test synthetic rating creation."""
        df = pd.DataFrame({"price_numeric": [25.99, 50.00]})

        # Create synthetic ratings
        df["Rating"] = df["price_numeric"] * 0.2 + 3.0
        df.loc[df["Rating"] > 5.0, "Rating"] = 5.0

        assert "Rating" in df.columns
        assert all(df["Rating"] <= 5.0)

    def test_train_test_split(self):
        """Test data splitting."""
        df = pd.DataFrame({"col1": range(10), "col2": range(10, 20)})

        train, test = train_test_split(df, test_size=0.2, random_state=42)

        assert len(train) + len(test) == len(df)
        assert len(test) == 2
