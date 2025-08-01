"""Model validator component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def validate_model_endpoint(
    endpoint: Input[Model],
    validation_output: Output[Model],
    project: str,
    region: str,
    test_instances: list = None,
    expected_prediction_count: int = 1,
):
    """Validates a deployed model by sending test predictions."""
    from google.cloud import aiplatform
    import logging

    if test_instances is None:
        test_instances = [[3.99, "Italy", "Tuscany", "White", "Light", "Pinot Grigio"]]

    try:
        # Initialize AI Platform
        aiplatform.init(project=project, location=region)
        endpoint_resource_name = endpoint.uri

        if not endpoint_resource_name:
            raise ValueError("Endpoint resource_name is missing")

        # Get endpoint and make prediction
        endpoint_instance = aiplatform.Endpoint(endpoint_resource_name)
        logging.info("Retrieved endpoint: %s", endpoint_instance.resource_name)

        prediction_response = endpoint_instance.predict(instances=test_instances)
        logging.info("Prediction response received: %s", prediction_response)

        # Validate response structure
        if not hasattr(prediction_response, "predictions"):
            raise ValueError("Response missing 'predictions' attribute")

        predictions = prediction_response.predictions

        if not isinstance(predictions, list):
            raise ValueError("Predictions should be a list")

        if len(predictions) != expected_prediction_count:
            raise ValueError(
                f"Expected predictions count : {expected_prediction_count}"
            )

        # Log success and store results
        logging.info("Model validation successful!")
        logging.info("Received %s predictions as expected", len(predictions))
        logging.info("Sample prediction: %s", predictions[0] if predictions else "None")

        validation_results = {
            "status": "PASSED",
            "endpoint": endpoint_resource_name,
            "test_instances_count": len(test_instances),
            "predictions_count": len(predictions),
            "sample_prediction": predictions[0] if predictions else None,
        }
        validation_output.uri = endpoint_resource_name
        validation_output.metadata = validation_results

        logging.info(
            "Model validation completed successfully for endpoint: %s",
            endpoint_resource_name,
        )
    except Exception as exc:
        logging.error("Model validation failed: %s", exc)
        validation_results = {
            "status": "FAILED",
            "endpoint": endpoint.uri or "unknown",
            "error": str(exc),
        }
        validation_output.metadata = validation_results
        raise
