"""Model evaluator component for wine quality pipeline - UTF-8 safe minimal version."""

from kfp.v2.dsl import Dataset, Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image=BASE_CONTAINER_IMAGE,
)
def evaluate_model(
    model_artifact: Input[Model], 
    test_data: Input[Dataset],
    evaluated_model: Output[Model]
) -> float:
    """Evaluates the wine rating prediction model - minimal version to avoid UTF-8 issues."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import joblib
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    try:
        logging.info("Starting model evaluation")

        # Load model
        model_path = model_artifact.path + ".joblib"
        logging.info("Loading trained model")
        try:
            with open(model_path, "rb") as file:
                model = joblib.load(file)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise

        # Load test data
        test_df = pd.read_csv(test_data.path)
        logging.info(
            "Loaded test data: %s rows, %s columns", test_df.shape[0], test_df.shape[1]
        )

        # Get features from metadata or use defaults
        features_str = model_artifact.metadata.get("features", "")
        if features_str:
            import ast
            features = ast.literal_eval(features_str)
        else:
            categorical_features = ["Country", "Region", "Type", "Style", "Grape"]
            categorical_features = [
                col for col in categorical_features if col in test_df.columns
            ]
            numeric_features = ["price_numeric"]
            numeric_features = [
                col for col in numeric_features if col in test_df.columns
            ]
            features = numeric_features + categorical_features

        target = model_artifact.metadata.get("target", "Rating")
        logging.info("Using %s features for evaluation", len(features))

        # Make predictions
        test_features = test_df[features]
        test_target = test_df[target]
        predictions = model.predict(test_features)

        # Calculate metrics
        mse = mean_squared_error(test_target, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_target, predictions)
        r2 = r2_score(test_target, predictions)

        logging.info(
            "Evaluation Results - RMSE: %.4f, MAE: %.4f, R²: %.4f", rmse, mae, r2
        )

        # Calculate quality score (higher = better)
        r2_norm = max(0, r2)
        rmse_score = max(0, 1.0 - rmse)
        quality_score = (0.7 * r2_norm) + (0.3 * rmse_score)

        logging.info("Model quality score: %.4f", quality_score)

        # MINIMAL metadata transfer - only copy essential info to avoid UTF-8 issues
        evaluated_model.uri = model_artifact.uri
        evaluated_model.metadata = model_artifact.metadata.copy()
        
        # Add only essential evaluation metrics as strings (safer than complex objects)
        evaluated_model.metadata["quality_score"] = str(round(quality_score, 4))
        evaluated_model.metadata["r2_score"] = str(round(r2, 4))
        evaluated_model.metadata["rmse"] = str(round(rmse, 4))
        evaluated_model.metadata["mae"] = str(round(mae, 4))
        evaluated_model.metadata["mse"] = str(round(mse, 4))
        evaluated_model.metadata["evaluation_status"] = "completed"

        logging.info("Model evaluation completed successfully")
        return quality_score

    except Exception as e:
        logging.error("Model evaluation failed: %s", e)
        raise