"""Data preprocessor component for wine quality pipeline."""

from kfp.v2.dsl import component, Input, Output, Dataset
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image=BASE_CONTAINER_IMAGE,
)
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Preprocesses the wine data for the prediction model."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import re
    import logging
    from sklearn.model_selection import train_test_split

    try:
        logging.info("Starting data preprocessing")

        # Load data
        df = pd.read_csv(input_data.path)
        logging.info("Loaded data: %d rows, %d columns", df.shape[0], df.shape[1])

        # Data cleanup (handle missing columns & handle null price)
        categorical_features = ["Country", "Region", "Type", "Style", "Grape"]
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # Process price column
        if "Price" in df.columns:
            df["price_numeric"] = df["Price"].apply(
                lambda x: (
                    float(re.search(r"(\d+\.?\d*)", str(x)).group(1))
                    if pd.notna(x) and re.search(r"\d+\.?\d*", str(x))
                    else 0
                )
            )
            logging.info("Processed Price column to numeric values")
        else:
            df["price_numeric"] = 0
            logging.warning("Price column not found, using default value 0")

        # Target column (rating) for final prediction
        if "Rating" not in df.columns:
            df["Rating"] = df["price_numeric"] * 0.2 + 3.0
            df.loc[df["Rating"] > 5.0, "Rating"] = 5.0
            logging.warning("Created synthetic Rating column based on price")
        else:
            logging.info("Using existing Rating column")

        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        logging.info(
            "Train: %d rows, Test: %d rows", train_df.shape[0], test_df.shape[0]
        )

        # Save datasets
        train_df.to_csv(train_data.path, index=False)
        test_df.to_csv(test_data.path, index=False)
        df.to_csv(output_data.path, index=False)

        logging.info("Data preprocessing completed successfully")

    except Exception as e:
        logging.error("Data loading failed: %s", e)
        raise
