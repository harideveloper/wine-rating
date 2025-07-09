# run_rating_prediction_pipeline.py
"""
Run the rating prediction batch pipeline on Vertex AI.
Updated for direct batch prediction without model registry.
"""
from google.cloud import aiplatform
from rating_prediction_pipeline import rating_prediction_pipeline, compile_pipeline
from rating_prediction_constants import (
    PROJECT_ID,
    REGION,
    DATA_PATH,
    BATCH_INPUT_PATH,
    BATCH_OUTPUT_PATH,
    PIPELINE_ROOT,
    PIPELINE_FILE,
    PIPELINE_JOB_DISPLAY_NAME,
    MODEL_DISPLAY_NAME,
    GCS_BUCKET,
    ENABLE_DETAILED_LOGGING
)
import logging
from typing import Optional


def setup_logging():
    """Setup logging configuration."""
    if ENABLE_DETAILED_LOGGING:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    return logging.getLogger(__name__)


def validate_inputs(
    project_id: str,
    data_path: str,
    batch_input_path: str,
    batch_output_path: str
):
    """Validate input parameters."""
    logger = logging.getLogger(__name__)
    
    if not project_id:
        raise ValueError("Project ID cannot be empty")
    
    if not data_path.startswith('gs://'):
        raise ValueError("Data path must be a GCS URL (gs://...)")
    
    if not batch_input_path.startswith('gs://'):
        raise ValueError("Batch input path must be a GCS URL (gs://...)")
    
    if not batch_output_path.startswith('gs://'):
        raise ValueError("Batch output path must be a GCS URL (gs://...)")
    
    logger.info("✅ Input validation passed")


def run_rating_prediction_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    data_path: str = DATA_PATH,
    batch_input_path: str = BATCH_INPUT_PATH,
    batch_output_path: str = BATCH_OUTPUT_PATH,
    pipeline_root: str = PIPELINE_ROOT,
    pipeline_file: str = PIPELINE_FILE,
    model_name: str = MODEL_DISPLAY_NAME,
    gcs_bucket: str = GCS_BUCKET,
    enable_caching: bool = True,
    sync: bool = True
):
    """
    Run the rating prediction batch pipeline on Vertex AI.
    Updated for direct batch prediction without model registry.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        data_path: Path to training data in GCS
        batch_input_path: Path to batch input data in GCS
        batch_output_path: Path to store predictions in GCS
        pipeline_root: Root path for pipeline artifacts
        pipeline_file: Name of compiled pipeline file
        model_name: Display name for the model
        gcs_bucket: GCS bucket name for storing models
        enable_caching: Whether to enable pipeline caching
        sync: Whether to run pipeline synchronously
    
    Returns:
        PipelineJob: The completed pipeline job
    """
    logger = setup_logging()
    
    try:
        # Validate inputs
        validate_inputs(project_id, data_path, batch_input_path, batch_output_path)
        
        # Initialize Vertex AI
        logger.info(f"INFO: Initializing Vertex AI...")
        logger.info(f"INFO: Project: {project_id}")
        logger.info(f"INFO: Region: {region}")
        
        aiplatform.init(project=project_id, location=region)
        
        # Compile pipeline
        logger.info(f"INFO: Compiling pipeline...")
        compile_pipeline(pipeline_file)
        logger.info(f"INFO: Pipeline compiled to {pipeline_file}")
        
        # Create pipeline job
        job = aiplatform.PipelineJob(
            display_name=PIPELINE_JOB_DISPLAY_NAME,
            template_path=pipeline_file,
            pipeline_root=pipeline_root,
            parameter_values={
                "data_path": data_path,
                "batch_input_path": batch_input_path,
                "batch_output_path": batch_output_path,
                "model_name": model_name,
                "gcs_bucket": gcs_bucket
            },
            enable_caching=enable_caching
        )
        
        # Log pipeline details
        logger.info(f"DEBUG: Pipeline Configuration:")
        logger.info(f"DEBUG: Training data: {data_path}")
        logger.info(f"DEBUG: Batch input: {batch_input_path}")
        logger.info(f"DEBUG: Output folder: {batch_output_path}")
        logger.info(f"DEBUG: Model name: {model_name}")
        logger.info(f"DEBUG: GCS bucket: {gcs_bucket}")
        logger.info(f"DEBUG: Caching enabled: {enable_caching}")
        logger.info(f"DEBUG: Pipeline type: Direct batch prediction (no registry)")
        
        logger.info(f"INFO: Starting pipeline execution...")
        job.run(sync=sync)
        
        if sync:
            logger.info(f"INFO: Pipeline completed!")
            logger.info(f"INFO: Job name: {job.display_name}")
            logger.info(f"INFO: Job ID: {job.resource_name}")
            logger.info(f"INFO: Job state: {job.state}")
            
            if job.state == "PIPELINE_STATE_SUCCEEDED":
                logger.info(f"INFO: Pipeline succeeded! Check your GCS bucket for predictions.")
                logger.info(f"INFO: Predictions location: {batch_output_path}/predictions.jsonl")
                logger.info(f"INFO: Model saved to: gs://{gcs_bucket}/models/{model_name}/model.pkl")
            else:
                logger.error(f"ERROR: Pipeline failed with state: {job.state}")
                logger.info(f"INFO: Check the Vertex AI console for detailed error logs")
        else:
            logger.info(f"INFO: Pipeline started asynchronously")
            logger.info(f"INFO: Job ID: {job.resource_name}")
        
        return job
        
    except Exception as e:
        logger.error(f"ERROR: Pipeline execution failed: {str(e)}")
        raise


def check_pipeline_status(job_resource_name: str, project_id: str = PROJECT_ID, region: str = REGION):
    """
    Check the status of a running pipeline job.
    
    Args:
        job_resource_name: Resource name of the pipeline job
        project_id: GCP project ID
        region: GCP region
    """
    logger = setup_logging()
    
    aiplatform.init(project=project_id, location=region)
    
    try:
        job = aiplatform.PipelineJob.get(job_resource_name)
        logger.info(f"INFO: Job Name = {job.display_name}")
        logger.info(f"INFO: Job State = {job.state}")
        logger.info(f"INFO: Job Created Time : {job.create_time}")
        
        if job.end_time:
            logger.info(f"INFO: Job Completed: {job.end_time}")
        
        return job
    except Exception as e:
        logger.error(f"ERROR: Failed to get job status: {str(e)}")
        raise


def create_sample_batch_input():
    """
    Create a sample batch input file for testing.
    This is a helper function to generate test data.
    """
    logger = setup_logging()
    import json
    from google.cloud import storage
    
    # Sample data matching the wine dataset structure
    sample_instances = [
        {
            "Country": "France",
            "Type": "Red",
            "Grape": "Pinot Noir",
            "Style": "Dry",
            "Price": "$25.99"
        },
        {
            "Country": "Italy",
            "Type": "White",
            "Grape": "Chardonnay",
            "Style": "Dry",
            "Price": "$18.50"
        },
        {
            "Country": "Spain",
            "Type": "Red",
            "Grape": "Tempranillo",
            "Style": "Medium",
            "Price": "$32.00"
        }
    ]
    
    # Create JSONL content
    jsonl_content = ""
    for instance in sample_instances:
        jsonl_content += json.dumps(instance) + "\n"
    
    # Upload to GCS
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("batch.jsonl")
        blob.upload_from_string(jsonl_content)
        
        logger.info(f"INFO: Sample batch input created at: gs://{GCS_BUCKET}/batch.jsonl")
        logger.info(f"INFO: Created {len(sample_instances)} sample instances")
        
    except Exception as e:
        logger.error(f"ERROR: Failed to create sample batch input: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Main execution block.
    Update the constants in rating_prediction_constants.py before running.
    """
    logger = setup_logging()
    
    logger.info("INFO: Starting Rating Prediction Pipeline (Direct Batch Prediction)...")
    
    try:
        # Optionally create sample batch input data
        # Uncomment the next line if you need to create test data
        # create_sample_batch_input()
        
        job = run_rating_prediction_pipeline()
        
        if job.state == "PIPELINE_STATE_SUCCEEDED":
            logger.info("INFO: Success! Your rating predictions are ready!")
            logger.info("INFO: Pipeline completed without using model registry or external containers.")
        else:
            logger.warning(f"WARNING: Pipeline completed with state: {job.state}")
            
    except Exception as e:
        logger.error(f"ERROR: Pipeline execution failed: {str(e)}")
        exit(1)