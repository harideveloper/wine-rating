"""
Run the simple wine rating prediction pipeline on Vertex AI.
"""
from google.cloud import aiplatform
from wine_rating_pipeline import wine_pipeline
from kfp.v2 import compiler


from constants import (
    PROJECT_ID,
    GCS_BUCKET,
    DATA_PATH,
    REGION,
    BATCH_INPUT_PATH,
    BATCH_OUTPUT_PATH,
    PIPELINE_ROOT,
    SERVING_CONTAINER_IMAGE_URI,
    MACHINE_TYPE,
    MIN_REPLICA_COUNT,
    MAX_REPLICA_COUNT,
)

def compile_pipeline(output_file: str = "wine_pipeline.json"):
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=wine_pipeline,
        package_path=output_file
    )
    print(f"✅ Pipeline compiled to {output_file}")


def run_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    data_path: str = DATA_PATH,
    batch_input_path: str = BATCH_INPUT_PATH,
    batch_output_path: str = BATCH_OUTPUT_PATH,
    pipeline_root: str = PIPELINE_ROOT,
    pipeline_file: str = "wine_pipeline.json",
    serving_container_image_uri: str = SERVING_CONTAINER_IMAGE_URI,
    machine_type: str = MACHINE_TYPE,
    min_replica_count: int = MIN_REPLICA_COUNT,
    max_replica_count: int = MAX_REPLICA_COUNT

):
    """Run the simple wine pipeline on Vertex AI."""
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Compile the pipeline
    compile_pipeline(pipeline_file)
    
    # Create and run the pipeline job
    job = aiplatform.PipelineJob(
        display_name="simple-wine-batch-pipeline",
        template_path=pipeline_file,
        pipeline_root=pipeline_root,
        parameter_values={
            "data_path": data_path,
            "batch_input_path": batch_input_path,
            "batch_output_path": batch_output_path,
            "project": project_id,
            "region": region,
            "serving_container_image_uri": serving_container_image_uri,
            "machine_type": machine_type,
            "min_replica_count": min_replica_count,
            "max_replica_count": max_replica_count
        },
        enable_caching=True
    )
    
    print(f"🚀 Starting pipeline job...")
    print(f"📊 Training data: {data_path}")
    print(f"📝 Batch input: {batch_input_path}")
    print(f"📁 Output folder: {batch_output_path}")
    
    # Run the pipeline
    job.run(sync=True)
    
    print(f"✅ Pipeline completed!")
    print(f"📋 Job name: {job.display_name}")
    print(f"🆔 Job ID: {job.resource_name}")
    print(f"📈 Job state: {job.state}")
    
    return job


if __name__ == "__main__":    
    job = run_pipeline(
        project_id=PROJECT_ID,
        data_path=DATA_PATH,
        batch_input_path=BATCH_INPUT_PATH,
        batch_output_path=BATCH_OUTPUT_PATH,
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        machine_type=MACHINE_TYPE,
        min_replica_count=MIN_REPLICA_COUNT,
        max_replica_count=MAX_REPLICA_COUNT
    )
    
    if job.state == "PIPELINE_STATE_SUCCEEDED":
        print("🎉 Pipeline succeeded! Check your GCS bucket for predictions.")
    else:
        print(f"❌ Pipeline failed with state: {job.state}")