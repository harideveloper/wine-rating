#!/usr/bin/env python
# Wine Recommendation Pipeline - Deployment Script

import argparse
import logging
from google.cloud import aiplatform, storage
from wine_recommendation_pipeline import compile_pipeline, run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wine-pipeline-deployment')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deploy Wine Recommendation Pipeline')

    parser.add_argument('--project-id', required=True,
                        help='GCP Project ID')
    parser.add_argument('--bucket', required=True,
                        help='GCS Bucket name (without gs:// prefix)')
    parser.add_argument('--region', default='europe-west2',
                        help='GCP Region (default: europe-west2)')
    parser.add_argument('--data-file', default='dataset/wine_v1.csv',
                        help='Path to wine data CSV in GCS bucket (default: dataset/wine_v1.csv)')
    parser.add_argument('--model-name', default='wine-recommendation-model',
                        help='Display name for the model (default: wine-recommendation-model)')
    parser.add_argument('--endpoint-name', default='wine-recommendation-endpoint',
                        help='Display name for the endpoint (default: wine-recommendation-endpoint)')
    parser.add_argument('--service-account',
                        help='Service account email for pipeline execution')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching for pipeline execution')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the pipeline, do not run it')

    return parser.parse_args()


def check_data_exists(project_id, bucket_name, data_file):
    """Check if the data file exists in GCS."""
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(data_file)

        if blob.exists():
            logger.info(f"✅ Data file found: gs://{bucket_name}/{data_file}")
            # Get file size
            blob.reload()
            size_mb = blob.size / (1024 * 1024)
            logger.info(f"   File size: {size_mb:.2f} MB")
            return True
        else:
            logger.error(
                f"❌ Data file not found: gs://{bucket_name}/{data_file}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking data file: {e}")
        return False


def check_bucket_permissions(project_id, bucket_name):
    """Check if we have permissions to write to the bucket."""
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

        # Try to create a test file
        test_blob = bucket.blob('pipeline_test_permission.txt')
        test_blob.upload_from_string('test')
        test_blob.delete()

        logger.info(
            f"✅ Write permissions confirmed for bucket: gs://{bucket_name}/")
        return True
    except Exception as e:
        logger.error(f"❌ Permission error for bucket gs://{bucket_name}/: {e}")
        return False


def check_vertex_ai_permissions(project_id, region):
    """Check if we have permissions to use Vertex AI."""
    try:
        aiplatform.init(project=project_id, location=region)
        # Just try to initialize the API
        logger.info(
            f"✅ Vertex AI permissions confirmed for project: {project_id}, region: {region}")
        return True
    except Exception as e:
        logger.error(f"❌ Vertex AI permission error: {e}")
        return False


def main():
    """Main function to deploy the wine recommendation pipeline."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("Wine Recommendation Pipeline Deployment")
    logger.info("=" * 80)
    logger.info(f"Project ID: {args.project_id}")
    logger.info(f"GCS Bucket: {args.bucket}")
    logger.info(f"Region: {args.region}")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Endpoint name: {args.endpoint_name}")
    logger.info("=" * 80)

    # Check prerequisites
    logger.info("Checking prerequisites...")

    # Check if data file exists
    if not check_data_exists(args.project_id, args.bucket, args.data_file):
        logger.error("Data file check failed. Aborting deployment.")
        return 1

    # Check bucket permissions
    if not check_bucket_permissions(args.project_id, args.bucket):
        logger.error("Bucket permission check failed. Aborting deployment.")
        return 1

    # Check Vertex AI permissions
    if not check_vertex_ai_permissions(args.project_id, args.region):
        logger.error("Vertex AI permission check failed. Aborting deployment.")
        return 1

    # Compile the pipeline
    logger.info("Compiling pipeline...")
    pipeline_file = "wine_recommendation_pipeline.json"
    compile_pipeline(pipeline_file)
    logger.info(f"✅ Pipeline compiled to {pipeline_file}")

    if args.compile_only:
        logger.info("Compilation complete. Skipping execution as requested.")
        return 0

    # Run the pipeline
    logger.info("Executing pipeline on Vertex AI...")
    try:
        data_path = f"gs://{args.bucket}/{args.data_file}"
        pipeline_root = f"gs://{args.bucket}/pipeline_root/wine_recommendation"

        job = run_pipeline(
            project_id=args.project_id,
            gcs_bucket=args.bucket,
            data_path=data_path,
            model_display_name=args.model_name,
            endpoint_display_name=args.endpoint_name,
            region=args.region,
            pipeline_root=pipeline_root,
            evaluation_threshold=0.5,
            # service_account=args.service_account,
        )

        logger.info(f"✅ Pipeline job submitted successfully")
        logger.info(f"Pipeline job name: {job.name}")
        logger.info(f"Pipeline job ID: {job.resource_name}")
        logger.info(
            f"Monitor progress at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={args.project_id}")
        return 0

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
