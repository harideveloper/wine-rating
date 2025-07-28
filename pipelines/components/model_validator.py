"""Model validator component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def validate_model(
    endpoint: Input[Model],
    validation_output: Output[Model],
    project: str,
    region: str,
    test_instances: list = None,
    expected_prediction_count: int = 1,
):
    """Validates a deployed model by sending test predictions."""
    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    import logging

    if test_instances is None:
        test_instances = [[3.99, "Italy", "Tuscany", "White", "Light", "Pinot Grigio"]]
    aiplatform.init(project=project, location=region)
    endpoint_resource_name = endpoint.uri
    if not endpoint_resource_name:
        logging.error("Endpoint resource_name not found in input")
        raise ValueError("ERROR: Endpoint resource_name is missing.")

    try:
        endpoint_instance = aiplatform.Endpoint(endpoint_resource_name)
        logging.info("Retrieved endpoint: %s ", endpoint_instance.resource_name)
    except Exception as e:
        logging.error("Error retrieving endpoint: %s", e)
        raise e

    try:
        prediction_response = endpoint_instance.predict(instances=test_instances)
        logging.info("Prediction response received: %s ", prediction_response)
        if not hasattr(prediction_response, "predictions"):
            logging.error("Response missing 'predictions' attribute")
            raise ValueError("Invalid prediction response structure")
        predictions = prediction_response.predictions
        if not isinstance(predictions, list):
            logging.error("Predictions is not a list")
            raise ValueError("Predictions should be a list")
        if len(predictions) != expected_prediction_count:
            logging.error(
                "Expected predictions count : %s", expected_prediction_count
            )  # pylint: disable=line-too-long
            raise ValueError( # pylint: disable=line-too-long, raising-format-tuple
                "Expected predictions count : %s", expected_prediction_count
            )  

        # Log validation results
        logging.info("Model validation successful!")
        logging.info("Received: %s predictions as expected ", len(predictions))
        logging.info(
            "Sample prediction: : %s ", predictions[0] if predictions else "None"
        )

        # Store validation results
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
        )  # pylint: disable=line-too-long
    except Exception as e:
        logging.error("Model validation error: %s", e)
        validation_results = {
            "status": "FAILED",
            "endpoint": endpoint_resource_name,
            "error": str(e),
        }  # pylint: disable=line-too-long
        validation_output.metadata = validation_results
        raise e
