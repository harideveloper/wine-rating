"""Model evaluator component for wine quality pipeline"""

from kfp.v2.dsl import Dataset, Model, Input, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image=BASE_CONTAINER_IMAGE,
)
# pylint: disable=import-outside-toplevel, too-many-arguments, too-many-statements, broad-exception-caught, too-many-positional-arguments, too-many-locals
def evaluate_model(
    trained_model: Input[Model],
    test_data: Input[Dataset],
    evaluated_model: Output[Model],
) -> float:
    """Evaluates the wine rating prediction model."""
    import joblib
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    try:
        logging.info("Starting model evaluation")

        # Load model
        model_path = trained_model.path + ".joblib"
        logging.info("Loading trained model from: %s", model_path)
        try:
            with open(model_path, "rb") as file:
                model = joblib.load(file)
            logging.info("Model loaded successfully")
        except FileNotFoundError:
            logging.error("Model file not found: %s", model_path)
            raise
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise

        # Load test data
        test_df = pd.read_csv(test_data.path)
        logging.info(
            "Loaded test data: %s rows, %s columns", test_df.shape[0], test_df.shape[1]
        )

        # Validate test data is not empty
        if test_df.empty:
            raise ValueError("Test dataset is empty")

        # Get features - use feature_order from trainer metadata (not 'features')
        feature_order_str = trained_model.metadata.get("feature_order", "")
        if feature_order_str:
            import ast

            features = ast.literal_eval(feature_order_str)
            logging.info("Using features from model metadata: %s", features)
        else:
            # Fallback to manual feature selection
            logging.warning("No feature_order in metadata, using fallback selection")
            categorical_features = ["Country", "Region", "Type", "Style", "Grape"]
            categorical_features = [
                col for col in categorical_features if col in test_df.columns
            ]
            numeric_features = ["price_numeric"]
            numeric_features = [
                col for col in numeric_features if col in test_df.columns
            ]
            features = numeric_features + categorical_features

        target = trained_model.metadata.get("target", "Rating")
        logging.info(
            "Using %s features for evaluation, target: %s", len(features), target
        )

        # Validate required columns exist
        missing_features = [f for f in features if f not in test_df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features in test data: {missing_features}"
            )

        if target not in test_df.columns:
            raise ValueError(f"Target column '{target}' not found in test data")

        # Make predictions
        test_features = test_df[features]
        test_target = test_df[target]

        # Validate no missing values in features/target
        if test_features.isnull().any().any():
            logging.warning(
                "Found null values in test features, this may affect predictions"
            )
        if test_target.isnull().any():
            raise ValueError("Target column contains null values")

        predictions = model.predict(test_features)
        logging.info("Generated %s predictions", len(predictions))

        # Calculate metrics
        mse = mean_squared_error(test_target, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_target, predictions)
        r2 = r2_score(test_target, predictions)

        logging.info(
            "Evaluation Results - RMSE: %.4f, MAE: %.4f, R²: %.4f", rmse, mae, r2
        )

        # Improved quality score calculation with validation
        r2_norm = max(0, min(1, r2))  # Clamp R² between 0 and 1

        # Better RMSE normalization (assuming wine ratings are 0-5 scale)
        max_rating = 5.0
        rmse_normalized = rmse / max_rating  # Normalize RMSE to 0-1 scale
        rmse_score = max(0, 1.0 - rmse_normalized)

        quality_score = (0.7 * r2_norm) + (0.3 * rmse_score)
        quality_score = max(0, min(1, quality_score))  # Ensure score is between 0-1

        logging.info(
            "Model quality score: %.4f (R²: %.4f, RMSE Score: %.4f)",
            quality_score,
            r2_norm,
            rmse_score,
        )

        # Validate quality score makes sense
        if quality_score < 0 or quality_score > 1:
            logging.warning(
                "Quality score %.4f is outside expected range [0,1]", quality_score
            )

        # Set evaluation metrics to model metadata
        evaluated_model.uri = trained_model.uri
        evaluated_model.metadata = trained_model.metadata.copy()
        evaluated_model.metadata["quality_score"] = str(round(quality_score, 4))
        evaluated_model.metadata["r2_score"] = str(round(r2, 4))
        evaluated_model.metadata["rmse"] = str(round(rmse, 4))
        evaluated_model.metadata["mae"] = str(round(mae, 4))
        evaluated_model.metadata["mse"] = str(round(mse, 4))
        evaluated_model.metadata["rmse_normalized"] = str(round(rmse_normalized, 4))
        evaluated_model.metadata["evaluation_status"] = "completed"
        evaluated_model.metadata["test_samples"] = str(len(test_df))

        logging.info("Model evaluation completed successfully")
        return quality_score
    except Exception as e:
        logging.error("Model evaluation failed: %s", e)
        # Set failure metadata
        if "evaluated_model" in locals():
            evaluated_model.metadata = (
                trained_model.metadata.copy()
                if hasattr(trained_model, "metadata")
                else {}
            )
            evaluated_model.metadata["evaluation_status"] = "failed"
            evaluated_model.metadata["error_message"] = str(e)
        raise
