# run_wine_quality_batch_predictor.py
"""
Run the Wine Quality Batch Predictor pipeline on Vertex AI.
"""
from google.cloud import aiplatform
from wine_quality_batch_predictor import wine_quality_batch_predictor, compile_pipeline
from constants import (
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
    """Setup standardized logging configuration."""
    log_level = logging.DEBUG if ENABLE_DETAILED_LOGGING else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_inputs(project_id: str, data_path: str, batch_input_path: str, batch_output_path: str):
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
    
    logger.debug("Input validation passed")


def run_wine_quality_predictor(
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
    Run the Wine Quality Batch Predictor pipeline on Vertex AI.
    
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
        logger.info("Initializing Vertex AI")
        logger.debug(f"Project: {project_id}, Region: {region}")
        aiplatform.init(project=project_id, location=region)
        
        # Compile pipeline
        logger.info("Compiling Wine Quality Batch Predictor")
        compile_pipeline(pipeline_file)
        logger.debug(f"Pipeline compiled to {pipeline_file}")
        
        # Log configuration
        logger.debug("Pipeline Configuration:")
        logger.debug(f"  Training data: {data_path}")
        logger.debug(f"  Batch input: {batch_input_path}")
        logger.debug(f"  Output folder: {batch_output_path}")
        logger.debug(f"  Model name: {model_name}")
        logger.debug(f"  GCS bucket: {gcs_bucket}")
        logger.debug(f"  Caching enabled: {enable_caching}")
        
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
        
        # Run pipeline
        logger.info("Starting Wine Quality pipeline execution")
        job.run(sync=sync)
        
        if sync:
            logger.info("Wine Quality pipeline completed")
            logger.info(f"Job name: {job.display_name}")
            logger.debug(f"Job ID: {job.resource_name}")
            
            if job.state == "PIPELINE_STATE_SUCCEEDED":
                logger.info("Pipeline succeeded")
                logger.info(f"Predictions saved to: {batch_output_path}")
                logger.info(f"Model saved to: gs://{gcs_bucket}/models/{model_name}/")
            else:
                logger.error(f"Pipeline failed with state: {job.state}")
                logger.error("Check Vertex AI console for detailed error logs")
        else:
            logger.info("Pipeline started asynchronously")
            logger.debug(f"Job ID: {job.resource_name}")
        
        return job
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


def check_pipeline_status(job_resource_name: str, project_id: str = PROJECT_ID, region: str = REGION):
    """
    Check the status of a running Wine Quality pipeline job.
    
    Args:
        job_resource_name: Resource name of the pipeline job
        project_id: GCP project ID
        region: GCP region
    """
    logger = setup_logging()
    aiplatform.init(project=project_id, location=region)
    
    try:
        job = aiplatform.PipelineJob.get(job_resource_name)
        logger.info(f"Job: {job.display_name}")
        logger.info(f"State: {job.state}")
        logger.info(f"Created: {job.create_time}")
        
        if job.end_time:
            logger.info(f"Completed: {job.end_time}")
        
        return job
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise


def create_sample_wine_batch_input():
    """Create a sample batch input file for testing wine quality prediction."""
    logger = setup_logging()
    import json
    from google.cloud import storage
    
    # Sample wine data
    sample_wines = [
        {"Country": "France", "Type": "Red", "Grape": "Pinot Noir", "Style": "Elegant & Complex", "Price": "$35.99"},
        {"Country": "Italy", "Type": "White", "Grape": "Chardonnay", "Style": "Crisp & Fresh", "Price": "$22.50"},
        {"Country": "Spain", "Type": "Red", "Grape": "Tempranillo", "Style": "Bold & Spicy", "Price": "$18.00"},
        {"Country": "Germany", "Type": "White", "Grape": "Riesling", "Style": "Sweet & Aromatic", "Price": "$28.75"},
        {"Country": "USA", "Type": "Sparkling", "Grape": "Chardonnay", "Style": "Rich & Full", "Price": "$45.00"}
    ]
    
    # Create JSONL content
    jsonl_content = "\n".join(json.dumps(wine) for wine in sample_wines) + "\n"
    
    # Upload to GCS
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("batch.jsonl")
        blob.upload_from_string(jsonl_content)
        
        logger.info(f"Sample batch input created: gs://{GCS_BUCKET}/batch.jsonl")
        logger.debug(f"Created {len(sample_wines)} sample wine records")
        
    except Exception as e:
        logger.error(f"Failed to create sample batch input: {str(e)}")
        raise


if __name__ == "__main__":
    """Main execution block."""
    logger = setup_logging()
    
    logger.info("Starting Wine Quality Batch Predictor Pipeline")
    
    try:
        # Optionally create sample batch input data
        # create_sample_wine_batch_input()
        
        job = run_wine_quality_predictor()
        
        if job.state == "PIPELINE_STATE_SUCCEEDED":
            logger.info("Success! Wine quality predictions are ready")
        else:
            logger.warning(f"Pipeline completed with state: {job.state}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        exit(1)