# run_wine_rating_pipeline.py
from google.cloud import aiplatform
from wine_rating_pipeline import compile_pipeline
from typing import Optional


def run_wine_rating_pipeline(
    project_id: str,
    gcs_bucket: str,
    data_path: str,
    model_display_name: str = "wine-rating-model",
    endpoint_display_name: str = "wine-rating-endpoint",
    region: str = "europe-west2",
    pipeline_root: Optional[str] = None,
    evaluation_threshold: float = 0.6
):
    """Run the wine rating prediction pipeline on Vertex AI."""
    aiplatform.init(project=project_id, location=region)

    if pipeline_root is None:
        pipeline_root = f"gs://{gcs_bucket}/pipeline_root/wine_rating"

    pipeline_file = "/pipeline/wine_rating_pipeline.json"
    compile_pipeline(pipeline_file)

    job = aiplatform.PipelineJob(
        display_name="wine-rating-job",
        template_path=pipeline_file,
        pipeline_root=pipeline_root,
        parameter_values={
            "data_path": data_path,
            "model_display_name": model_display_name,
            "endpoint_display_name": endpoint_display_name,
            "project": project_id,
            "region": region,
            "evaluation_threshold": evaluation_threshold,
            "batch_input_uri": f"gs://{gcs_bucket}/batch-inputs/wine_input.csv",
            "batch_output_uri": f"gs://{gcs_bucket}/batch-outputs/"
        },
        enable_caching=True
    )

    job.run(sync=True)

    print(f"Pipeline job launched: {job.display_name}")
    print(f"Pipeline job ID: {job.resource_name}")

    return job


if __name__ == "__main__":
    PROJECT_ID = "dev2-ea8f"
    GCS_BUCKET = "model-output-wine-dev2-ea8f"
    DATA_PATH = "gs://model-output-wine-dev2-ea8f/wine_data.csv"
    REGION = "europe-west2"

    job = run_wine_rating_pipeline(
        project_id=PROJECT_ID,
        gcs_bucket=GCS_BUCKET,
        data_path=DATA_PATH,
        model_display_name="wine-rating-model",
        endpoint_display_name="wine-rating-endpoint",
        region=REGION,
        evaluation_threshold=0.6
    )

    print(f"Pipeline job completed with state: {job.state}")
