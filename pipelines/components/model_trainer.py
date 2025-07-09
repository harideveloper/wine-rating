"""Model trainer component for wine quality pipeline."""

from kfp.v2.dsl import Dataset, Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image=BASE_CONTAINER_IMAGE,
)
def train_model(
    train_data: Input[Dataset],
    output_model: Output[Model],
    n_estimators: int,
    random_state: int,
):
    """Trains a wine rating prediction model using RandomForestRegressor."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import pandas as pd
    import joblib
    import logging
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    try:
        logging.info("Starting model training")

        # Load training data
        df = pd.read_csv(train_data.path)
        logging.info(
            "Loaded training data: %s rows, %s columns", df.shape[0], df.shape[1]
        )

        # Define features
        categorical_features = ["Country", "Region", "Type", "Style", "Grape"]
        categorical_features = [
            col for col in categorical_features if col in df.columns
        ]
        numeric_features = ["price_numeric"]
        numeric_features = [col for col in numeric_features if col in df.columns]
        feature_order = numeric_features + categorical_features
        target = "Rating"

        logging.info(
            "Features - Numeric: %s, Categorical: %s",
            len(numeric_features),
            len(categorical_features),
        )

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    [
                        i
                        for i, col in enumerate(feature_order)
                        if col in numeric_features
                    ],
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    [
                        i
                        for i, col in enumerate(feature_order)
                        if col in categorical_features
                    ],
                ),
            ],
            remainder="passthrough",
        )

        # model pipeline
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators, random_state=random_state
                    ),
                ),
            ]
        )

        # Prepare training data
        features_data = df[feature_order].values
        target_data = df[target].values
        logging.info("Training RandomForest with %s estimators", n_estimators)

        # Train model
        model_pipeline.fit(features_data, target_data)
        logging.info("Model training completed successfully")

        # Save model
        file_name = output_model.path + ".joblib"
        with open(file_name, "wb") as file:
            joblib.dump(model_pipeline, file)

        # Set model metadata
        output_model.metadata["framework"] = "sklearn"
        output_model.metadata["feature_order"] = str(feature_order)
        output_model.metadata["target"] = target
        logging.info("Model saved successfully")

    except Exception as e:
        logging.error("Model training failed: %s", e)
        raise
