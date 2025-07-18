"""Model validator component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments, too-many-positional-arguments
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
        test_instances = [
            [
                39.99,
                "France",
                "Bordeaux",
                "Red",
                "Bold",
                "Cabernet Sauvignon"
            ]
        ]

    aiplatform.init(project=project, location=region)
    endpoint_resource_name = endpoint.uri
    
    if not endpoint_resource_name:
        logging.error("Endpoint resource_name not found in input")
        raise ValueError("ERROR: Endpoint resource_name is missing.")
    
    try:
        endpoint_instance = aiplatform.Endpoint(endpoint_resource_name)
        logging.debug(f"Retrieved endpoint: {endpoint_instance.resource_name}")
    except Exception as e:
        logging.error(f"Error retrieving endpoint: {e}")
        raise e

    try:        
        # Send prediction request
        prediction_response = endpoint_instance.predict(instances=test_instances)
        logging.debug(f"Prediction response received: {prediction_response}")
        
        # Validate response
        if not hasattr(prediction_response, 'predictions'):
            logging.error("Response missing 'predictions' attribute")
            raise ValueError("Invalid prediction response structure")
            
        predictions = prediction_response.predictions
        if not isinstance(predictions, list):
            logging.error("Predictions is not a list")
            raise ValueError("Predictions should be a list")
            
        if len(predictions) != expected_prediction_count:
            logging.error(f"Expected {expected_prediction_count} predictions, got {len(predictions)}")
            raise ValueError(f"Expected {expected_prediction_count} predictions, got {len(predictions)}")
            
        # Log validation results
        logging.info("Model validation successful!")
        logging.info(f"Received {len(predictions)} predictions as expected")
        logging.info(f"Sample prediction: {predictions[0] if predictions else 'None'}")
        
        # Store validation results
        validation_results = {
            "status": "PASSED",
            "endpoint": endpoint_resource_name,
            "test_instances_count": len(test_instances),
            "predictions_count": len(predictions),
            "sample_prediction": predictions[0] if predictions else None
        }
        
        validation_output.uri = endpoint_resource_name
        validation_output.metadata = validation_results
        
        logging.info(f"Model validation completed successfully for endpoint: {endpoint_resource_name}")
        
    except Exception as e:
        logging.error(f"Model validation error: {e}")
        validation_results = {
            "status": "FAILED",
            "endpoint": endpoint_resource_name,
            "error": str(e)
        }
        validation_output.metadata = validation_results
        raise e