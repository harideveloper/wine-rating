from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Dataset, Model, Input, Output, component
from kfp.v2 import compiler


@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "gcsfs==2025.3.2"])
def load_data(data_path: str, output_data: Output[Dataset]):
    """Loads wine data from a CSV file located in GCS."""
    import pandas as pd

    def load_data_from_gcs(gcs_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(gcs_path)
            print(f"✅ Loaded wine data from GCS with shape: {df.shape}")
            return df
        except Exception as e:
            raise FileNotFoundError(
                f"❌ Failed to load GCS file: {gcs_path}. Error: {e}")

    df = load_data_from_gcs(data_path)

    # Add ID column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    df.to_csv(output_data.path, index=False)
    print(f"✅ Wine data saved to {output_data.path}")


@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5"])
def preprocess_data(input_data: Input[Dataset], output_data: Output[Dataset]):
    """Preprocesses the wine data for the recommendation model."""
    import os
    import pandas as pd
    import re

    # Load the input data
    df = pd.read_csv(input_data.path)
    print(f"✅ Loaded input data with shape: {df.shape}")

    # Handle missing values
    for col in ["Characteristics", "Grape", "Style", "Type", "Country", "Region"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Create a combined features column for similarity calculation
    df["combined_features"] = ""

    if "Characteristics" in df.columns:
        df["combined_features"] += df["Characteristics"] + " "

    if "Grape" in df.columns:
        df["combined_features"] += df["Grape"] + " "

    if "Style" in df.columns:
        df["combined_features"] += df["Style"] + " "

    if "Type" in df.columns:
        df["combined_features"] += df["Type"] + " "

    if "Country" in df.columns:
        df["combined_features"] += df["Country"] + " "

    if "Region" in df.columns:
        df["combined_features"] += df["Region"]

    # Clean up combined features
    df["combined_features"] = df["combined_features"].str.strip()

    # Extract numeric price values if available
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 0
        )

    # Write directly to the output path provided by the KFP framework
    # No need to create directories or join paths
    df.to_csv(output_data.path, index=False)
    print(f"✅ Preprocessed wine data saved at {output_data.path}")


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.6.1", "joblib==1.4.2"])
# def train_model(input_data: Input[Dataset], output_model: Output[Model]):
#     """Trains the wine recommendation model with minimal serialization (no raw pandas)."""
#     import pandas as pd
#     import numpy as np
#     import joblib
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Load CSV
#     df = pd.read_csv(input_data.path)
#     print(f"✅ Loaded data: {df.shape}")

#     # Vectorize features
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
#     print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")

#     # Cosine similarity matrix
#     similarity_matrix = cosine_similarity(tfidf_matrix)

#     # Save only essential pieces
#     model_data = {
#         "vectorizer": vectorizer,
#         "tfidf_matrix": tfidf_matrix,
#         "similarity_matrix": similarity_matrix,
#         "ids": df["id"].tolist(),
#         "names": df["Title"].tolist() if "Title" in df.columns else df["id"].astype(str).tolist(),
#         # Optionally, store other metadata in plain dict/list formats
#     }

#     joblib.dump(model_data, output_model.path)
#     print(f"✅ Lightweight model saved to {output_model.path}")

@component(
    packages_to_install=["pandas==1.5.3",
                         "numpy==1.23.5", "scikit-learn==1.6.1"],
    base_image="python:3.9"
)
def train_model(input_data: Input[Dataset], output_model: Output[Model]):
    """Trains the wine recommendation model."""
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the preprocessed data
    df = pd.read_csv(input_data.path)
    print(f"✅ Loaded preprocessed data with shape: {df.shape}")

    # Create a TF-IDF vectorizer for the text features
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    print(f"✅ Created TF-IDF matrix with shape: {tfidf_matrix.shape}")

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"✅ Created similarity matrix with shape: {similarity_matrix.shape}")

    # Extract only the essential data needed for inference
    ids = df['id'].values
    names = df['Title'].values if 'Title' in df.columns else df['id'].astype(
        str).values

    # Create a model object that doesn't depend on pandas
    model = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'similarity_matrix': similarity_matrix,
        'ids': ids,
        'names': names,
    }

    # Save the model using pickle
    file_name = output_model.path + ".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

    # Add metadata
    output_model.metadata["framework"] = "sklearn"

    print(f"✅ Model trained and saved to {file_name}")


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.6.1", "joblib==1.4.2"])
# def train_model(input_data: Input[Dataset], output_model: Output[Model]):
#     """Trains the wine recommendation model."""
#     import os
#     import pandas as pd
#     import numpy as np
#     import joblib
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Load the preprocessed data directly from the input path
#     df = pd.read_csv(input_data.path)
#     print(f"✅ Loaded preprocessed data with shape: {df.shape}")

#     # Create a TF-IDF vectorizer for the text features
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
#     print(f"✅ Created TF-IDF matrix with shape: {tfidf_matrix.shape}")

#     # Calculate similarity matrix
#     similarity_matrix = cosine_similarity(tfidf_matrix)
#     print(f"✅ Created similarity matrix with shape: {similarity_matrix.shape}")

#     # Create model data structure
#     model_data = {
#         'df': df,
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'similarity_matrix': similarity_matrix,
#         'ids': df['id'].values,
#         'names': df['Title'].values if 'Title' in df.columns else df['id'].astype(str).values
#     }

#     # Save the model directly to the output path
#     joblib.dump(model_data, output_model.path)
#     print(f"✅ Model trained and saved to {output_model.path}")


@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.6.1", "joblib==1.4.2"])
def evaluate_model(model_artifact: Input[Model]) -> float:
    """Evaluates the wine recommendation model."""
    import os
    import joblib

    print(f"Loading model from: {model_artifact.path}")

    # Load the model directly from the input artifact path
    try:
        model_data = joblib.load(model_artifact.path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # List directory contents for debugging
        model_dir = os.path.dirname(model_artifact.path)
        if os.path.exists(model_dir):
            print(f"Directory contents of {model_dir}:")
            print(os.listdir(model_dir))
        raise

    # Extract model components
    similarity_matrix = model_data['similarity_matrix']
    total_wines = similarity_matrix.shape[0]
    print(f"Evaluating model with {total_wines} wines...")

    # Calculate coverage (percentage of wines with strong recommendations)
    coverage = 0

    for i in range(total_wines):
        # Get top 5 recommendations for each wine (excluding itself)
        sim_scores = [(j, similarity_matrix[i, j])
                      for j in range(total_wines) if j != i]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]

        # If we have at least one recommendation with similarity > 0.5, count it as covered
        if any(score > 0.5 for _, score in sim_scores):
            coverage += 1

    coverage_score = coverage / total_wines
    print(f"✅ Evaluation completed. Quality score: {coverage_score:.2f}")
    return coverage_score


@component(packages_to_install=["joblib==1.4.2"])
def upload_model(model_artifact: Input[Model], uploaded_model_artifact: Output[Model]):
    """Uploads the wine recommendation model artifact."""
    import os
    import shutil

    # Create a directory structure for the model
    model_dir = os.path.dirname(uploaded_model_artifact.path)
    os.makedirs(model_dir, exist_ok=True)

    # Define the path where the model file will be saved
    model_file_path = os.path.join(model_dir, "model.joblib")

    # Copy the model file
    print(f"Copying model from {model_artifact.path} to {model_file_path}")
    shutil.copy(model_artifact.path, model_file_path)

    # Set the URI to the directory containing the model
    uploaded_model_artifact.uri = model_dir
    print(f"✅ Model uploaded to directory: {model_dir}")
    print(f"✅ Model URI set to: {uploaded_model_artifact.uri}")


@component(packages_to_install=["google-cloud-aiplatform==1.22.1", "joblib==1.4.2"])
def register_model(
    model_artifact: Input[Model],
    registered_model: Output[Model],
    model_display_name: str,
    serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest",
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
):
    """Registers the wine recommendation model to Vertex AI Model Registry."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    try:
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact.uri,
            serving_container_image_uri=serving_container_image_uri,
        )
        print(f"✅ Model registered: {model.resource_name}")

        # Store model information
        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise e


# @component(packages_to_install=["numpy==1.23.5", "pandas==1.5.3", "scikit-learn==1.6.1", "joblib==1.4.2", "google-cloud-aiplatform==1.12.1"])
# def run_wine_batch_prediction(
#     model_registry_name: Input[Model],
#     batch_prediction_job: Output[Artifact],
#     input_data_uri: str,
#     output_uri_prefix: str,
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2",
#     machine_type: str = "n1-standard-4"
# ):
#     """Runs a batch prediction job for the wine recommendation model."""
#     from google.cloud import aiplatform
#     import time

#     # Initialize Vertex AI
#     aiplatform.init(project=project, location=region)

#     # Get the model resource name
#     model_name = model_registry_name.metadata.get("resource_name")
#     if not model_name:
#         print(f"❌ Error: Model resource_name not found in metadata.")
#         raise ValueError("Model resource_name is missing from metadata.")

#     try:
#         model = aiplatform.Model(model_name)
#         print(f"✅ Retrieved model: {model.resource_name}")
#     except Exception as e:
#         print(f"❌ Error retrieving model: {e}")
#         raise e

#     batch_job_display_name = f"wine-rec-batch-{int(time.time())}"

#     try:
#         # Launch batch prediction job with proper packages
#         print(f"Starting batch prediction job...")

#         # Make sure to include all necessary packages
#         batch_prediction_job_obj = model.batch_predict(
#             job_display_name=batch_job_display_name,
#             gcs_source=input_data_uri,
#             gcs_destination_prefix=output_uri_prefix,
#             machine_type=machine_type,
#             starting_replica_count=1,
#             max_replica_count=2,
#             # Make sure to install pandas for the batch job
#             model_parameters={"content_type": "json"},
#             # # This is essential - include pandas in the batch job
#             # env_vars={"ADDITIONAL_PYTHON_MODULES": "pandas scikit-learn joblib"}
#         )

#         print(
#             f"✅ Batch prediction job started: {batch_prediction_job_obj.resource_name}")

#         # Set metadata
#         batch_prediction_job.metadata = {
#             "resource_name": batch_prediction_job_obj.resource_name,
#             "display_name": batch_job_display_name,
#             "model_name": model.resource_name,
#             "output_location": output_uri_prefix
#         }

#         # Monitor the job
#         print(f"Monitoring batch prediction job...")
#         batch_prediction_job_obj.wait()

#         # Check final status
#         print(
#             f"Batch prediction job completed with state: {batch_prediction_job_obj.state}")

#         if batch_prediction_job_obj.state == aiplatform.compat.types.job_state.JobState.JOB_STATE_SUCCEEDED:
#             print(f"✅ Batch prediction completed successfully!")
#             print(f"Output data available at: {output_uri_prefix}")
#         else:
#             print(f"❌ Batch prediction job failed or was cancelled.")
#             print(f"Final state: {batch_prediction_job_obj.state}")

#     except Exception as e:
#         print(f"❌ Batch prediction error: {e}")
#         raise e

@component(
    packages_to_install=["google-cloud-aiplatform>=1.22.1"],
    base_image="python:3.9"
)
def deploy_to_vertex_endpoint(
    model_registry_name: Input[Model],
    project: str,
    region: str,
    endpoint_display_name: str
):
    """Deploys a model to Vertex AI Endpoint."""
    from google.cloud import aiplatform

    # Initialize Vertex AI
    aiplatform.init(project=project, location=region)

    # Get the model resource name
    model_name = model_registry_name.metadata.get("resource_name")
    model = aiplatform.Model(model_name)
    print(f"✅ Retrieved model: {model.resource_name}")

    # Check if endpoint exists, otherwise create it
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc"
    )

    if endpoints:
        endpoint = endpoints[0]
        print(f"⚠️ Using existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )
        print(f"✅ Created new endpoint: {endpoint.resource_name}")

    # Deploy the model
    deployment = endpoint.deploy(
        model=model,
        traffic_split={"0": 100},
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )

    print(f"✅ Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint.resource_name


# @component(packages_to_install=["google-cloud-aiplatform==1.12.1"])
# def deploy_to_vertex_endpoint(
#     model_registry_name: Input[Model],
#     endpoint: Output[Model],
#     endpoint_display_name: str,
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2",
# ):
#     """Deploys the registered wine recommendation model to a Vertex AI Endpoint."""
#     from google.cloud import aiplatform

#     aiplatform.init(project=project, location=region)
#     model_name = model_registry_name.metadata.get("resource_name")
#     if not model_name:
#         print(f"❌ Error: Model resource_name not found in metadata.")
#         raise ValueError("Model resource_name is missing from metadata.")

#     try:
#         model = aiplatform.Model(model_name)
#         print(f"✅ Retrieved model: {model.resource_name}")
#     except Exception as e:
#         print(f"❌ Error retrieving model: {e}")
#         raise e

#     deployed_model_display_name = f"{model_registry_name.metadata.get('display_name', 'model')}-deployed"

#     try:
#         endpoints = aiplatform.Endpoint.list(
#             filter=f'display_name="{endpoint_display_name}"',
#             order_by="create_time desc",
#             project=project,
#             location=region,
#         )

#         if endpoints:
#             endpoint_to_use = endpoints[0]
#             print(
#                 f"⚠️ Using existing endpoint: {endpoint_to_use.resource_name}")
#         else:
#             endpoint_to_use = aiplatform.Endpoint.create(
#                 display_name=endpoint_display_name,
#                 sync=True,
#             )
#             print(f"✅ Created new endpoint: {endpoint_to_use.resource_name}")

#         deployed_model = endpoint_to_use.deploy(
#             model=model,
#             deployed_model_display_name=deployed_model_display_name,
#             traffic_split={"0": 100},
#             sync=True,
#         )

#         endpoint.uri = endpoint_to_use.resource_name
#         print(f"✅ Model deployed to endpoint: {endpoint_to_use.resource_name}")

#     except Exception as e:
#         print(f"❌ Deployment failed: {e}")
#         raise e


@component(packages_to_install=["google-cloud-aiplatform>=1.38.0"])
def deploy_to_vertex_endpoint(
    model_registry_name: Input[Model],
    endpoint: Output[Model],
    endpoint_display_name: str,
    project: str = "dev2-ea8f",
    region: str = "europe-west2"
):
    """Deploys a model to Vertex AI Endpoint without monitoring the operation."""
    from google.cloud import aiplatform
    import time

    # Initialize Vertex AI
    aiplatform.init(project=project, location=region)

    # Get the model resource name
    model_name = model_registry_name.metadata.get("resource_name")
    if not model_name:
        print(f"❌ Error: Model resource_name not found in metadata.")
        raise ValueError("Model resource_name is missing from metadata.")

    try:
        model = aiplatform.Model(model_name)
        print(f"✅ Retrieved model: {model.resource_name}")
    except Exception as e:
        print(f"❌ Error retrieving model: {e}")
        raise e

    deployed_model_display_name = f"{model_registry_name.metadata.get('display_name', 'model')}-deployed"

    try:
        # Check if endpoint already exists
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"',
            order_by="create_time desc",
            project=project,
            location=region,
        )

        if endpoints:
            endpoint_to_use = endpoints[0]
            print(
                f"⚠️ Using existing endpoint: {endpoint_to_use.resource_name}")
        else:
            endpoint_to_use = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                project=project,
                location=region
            )
            print(f"✅ Created new endpoint: {endpoint_to_use.resource_name}")

        # Set the endpoint URI immediately
        endpoint.uri = endpoint_to_use.resource_name
        print(f"✅ Endpoint URI set to: {endpoint_to_use.resource_name}")

        # Default traffic split - send all traffic to the new model
        traffic_split = {"0": 100}
        print(f"Using traffic split: {traffic_split}")

        # Start deployment
        print(f"Starting deployment of model to endpoint...")
        print(f"NOTE: Deployment may take 15-30 minutes to complete.")
        print(
            f"The pipeline will continue, but the endpoint might not be ready immediately.")

        try:
            # Deploy the model using the endpoint's deploy method directly
            # Don't even try to capture the return value
            endpoint_to_use.deploy(
                model=model,
                deployed_model_display_name=deployed_model_display_name,
                machine_type="n1-standard-2",
                min_replica_count=1,
                max_replica_count=1,
                traffic_split=traffic_split,
                sync=False  # Use async deployment
            )

            print(f"✅ Deployment initiated successfully.")
            print(
                f"⚠️ Deployment is continuing in the background and may take 15-30 minutes to complete.")

        except Exception as deploy_error:
            print(f"❌ Error during deployment: {deploy_error}")
            # Continue anyway since we've set the endpoint URI
            print(
                f"⚠️ Despite the error, you can check the endpoint in the Cloud Console.")

        # Provide info about how to check the status later
        print(f"You can check the deployment status later using the Google Cloud Console:")
        print(
            f"- Go to: https://console.cloud.google.com/vertex-ai/endpoints?project={project}")
        print(
            f"- Look for the endpoint with this resource name: {endpoint_to_use.resource_name}")
        print(f"Or using the Python API:")
        print(f"```")
        print(f"from google.cloud import aiplatform")
        print(f"aiplatform.init(project='{project}', location='{region}')")
        print(
            f"endpoint = aiplatform.Endpoint('{endpoint_to_use.resource_name}')")
        print(f"# Check status of models deployed to this endpoint")
        try:
            print(f"deployed_models = endpoint.list_models()")
        except:
            print(f"# If list_models() doesn't work, try:")
            print(f"# Check in the Google Cloud Console UI")
        print(f"```")

        print(f"⚠️ Note: Endpoint may still be deploying in the background.")

    except Exception as e:
        print(f"❌ General error: {e}")
        # Make sure endpoint URI is set even if there was an error
        if 'endpoint_to_use' in locals():
            endpoint.uri = endpoint_to_use.resource_name
            print(
                f"⚠️ Setting endpoint URI despite error: {endpoint_to_use.resource_name}")
        raise e


@dsl.pipeline(
    name="wine-recommendation-pipeline",
    description="End-to-end pipeline for training, evaluating and deploying a wine recommendation model."
)
def wine_pipeline(
    data_path: str,
    model_display_name: str = "wine-recommendation-model",
    endpoint_display_name: str = "wine-recommendation-endpoint",
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
    evaluation_threshold: float = 0.6
):
    # Load the wine data
    load_task = load_data(data_path=data_path)

    # Preprocess the data
    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"]
    )

    # Train the model
    train_task = train_model(
        input_data=preprocess_task.outputs["output_data"]
    )

    # Evaluate the model
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["output_model"]
    )
    # Use with_parameter to create a conditional branch
    with dsl.Condition(
        evaluate_task.output >= evaluation_threshold,
        name="check_model_quality"
    ):
        # Upload the model only if evaluation passes threshold
        upload_task = upload_model(
            model_artifact=train_task.outputs["output_model"]
        )

        # Register the model in Vertex AI Model Registry
        register_task = register_model(
            model_artifact=upload_task.outputs["uploaded_model_artifact"],
            model_display_name=model_display_name,
            project=project,
            region=region
        )

        batch_predict_task = run_wine_batch_prediction(
            model_registry_name=register_task.outputs["registered_model"],
            input_data_uri=f"gs://model-output-wine-dev2-ea8f/batch-inputs/wine_input.jsonl",
            output_uri_prefix=f"gs://model-output-wine-dev2-ea8f/batch-outputs/",
            project=project,
            region=region
        )

        # Deploy the model to a Vertex AI endpoint
        deploy_task = deploy_to_vertex_endpoint(
            model_registry_name=register_task.outputs["registered_model"],
            endpoint_display_name=endpoint_display_name,
            project=project,
            region=region
        )

# Compile the pipeline


def compile_pipeline(output_file="wine_recommendation_pipeline.json"):
    compiler.Compiler().compile(
        pipeline_func=wine_pipeline,
        package_path=output_file
    )


# Function to run the pipeline
def run_pipeline(
    project_id,
    gcs_bucket,
    data_path,
    model_display_name="wine-recommendation-model",
    endpoint_display_name="wine-recommendation-endpoint",
    region="europe-west2",
    pipeline_root=None,
    evaluation_threshold=0.5
):
    """Run the wine recommendation pipeline on Vertex AI."""
    from google.cloud import aiplatform

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Set pipeline root if not provided
    if pipeline_root is None:
        pipeline_root = f"gs://{gcs_bucket}/pipeline_root/wine_recommendation"

    # Compile the pipeline
    pipeline_file = "wine_recommendation_pipeline.json"
    compile_pipeline(pipeline_file)

    # Create a pipeline job
    job = aiplatform.PipelineJob(
        display_name="wine-recommendation-job",
        template_path=pipeline_file,
        pipeline_root=pipeline_root,
        parameter_values={
            "data_path": data_path,
            "model_display_name": model_display_name,
            "endpoint_display_name": endpoint_display_name,
            "project": project_id,
            "region": region
        },
        enable_caching=True
    )

    # Run the pipeline
    job.run(sync=True)

    return job


# from kfp.v2 import dsl
# from kfp.v2.dsl import Artifact, Dataset, Model, Input, Output, component
# from kfp.v2 import compiler


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "gcsfs==2025.3.2"])
# def load_data(data_path: str, output_data: Output[Dataset]):
#     """Loads wine data from a CSV file located in GCS."""
#     import pandas as pd

#     def load_data_from_gcs(gcs_path: str) -> pd.DataFrame:
#         try:
#             df = pd.read_csv(gcs_path)
#             print(f"✅ Loaded wine data from GCS with shape: {df.shape}")
#             return df
#         except Exception as e:
#             raise FileNotFoundError(
#                 f"❌ Failed to load GCS file: {gcs_path}. Error: {e}")

#     df = load_data_from_gcs(data_path)

#     # Add ID column if it doesn't exist
#     if 'id' not in df.columns:
#         df['id'] = range(1, len(df) + 1)

#     df.to_csv(output_data.path, index=False)
#     print(f"✅ Wine data saved to {output_data.path}")


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5"])
# def preprocess_data(input_data: Input[Dataset], output_data: Output[Dataset]):
#     """Preprocesses the wine data for the recommendation model."""
#     import os
#     import pandas as pd
#     import re

#     # Load the input data
#     df = pd.read_csv(input_data.path)
#     print(f"✅ Loaded input data with shape: {df.shape}")

#     # Handle missing values
#     for col in ["Characteristics", "Grape", "Style", "Type", "Country", "Region"]:
#         if col in df.columns:
#             df[col] = df[col].fillna("")

#     # Create a combined features column for similarity calculation
#     df["combined_features"] = ""

#     if "Characteristics" in df.columns:
#         df["combined_features"] += df["Characteristics"] + " "

#     if "Grape" in df.columns:
#         df["combined_features"] += df["Grape"] + " "

#     if "Style" in df.columns:
#         df["combined_features"] += df["Style"] + " "

#     if "Type" in df.columns:
#         df["combined_features"] += df["Type"] + " "

#     if "Country" in df.columns:
#         df["combined_features"] += df["Country"] + " "

#     if "Region" in df.columns:
#         df["combined_features"] += df["Region"]

#     # Clean up combined features
#     df["combined_features"] = df["combined_features"].str.strip()

#     # Extract numeric price values if available
#     if "Price" in df.columns:
#         df["price_numeric"] = df["Price"].apply(
#             lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
#             if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
#             else 0
#         )

#     # Write directly to the output path provided by the KFP framework
#     df.to_csv(output_data.path, index=False)
#     print(f"✅ Preprocessed wine data saved at {output_data.path}")


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.6.1", "scipy==1.10.1"],
#            base_image="python:3.9")
# def train_model(input_data: Input[Dataset], output_model: Output[Model]):
#     """Trains the wine recommendation model."""
#     import os
#     import pandas as pd
#     import numpy as np
#     import pickle
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Load the preprocessed data
#     df = pd.read_csv(input_data.path)
#     print(f"✅ Loaded preprocessed data with shape: {df.shape}")

#     # Create a TF-IDF vectorizer for the text features
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
#     print(f"✅ Created TF-IDF matrix with shape: {tfidf_matrix.shape}")

#     # Calculate similarity matrix
#     similarity_matrix = cosine_similarity(tfidf_matrix)
#     print(f"✅ Created similarity matrix with shape: {similarity_matrix.shape}")

#     # Extract only the essential data needed for inference
#     ids = df['id'].values
#     names = df['Title'].values if 'Title' in df.columns else df['id'].astype(
#         str).values

#     # Create a model object that doesn't depend on pandas
#     model = {
#         'vectorizer': vectorizer,
#         'tfidf_matrix': tfidf_matrix,
#         'similarity_matrix': similarity_matrix,
#         'ids': ids,
#         'names': names,
#     }

#     # Save the model using pickle
#     file_name = output_model.path + ".pkl"
#     with open(file_name, 'wb') as file:
#         pickle.dump(model, file)

#     # Add metadata
#     output_model.metadata["framework"] = "sklearn"

#     print(f"✅ Model trained and saved to {file_name}")


# @component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.6.1", "pickle5"])
# def evaluate_model(model_artifact: Input[Model]) -> float:
#     """Evaluates the wine recommendation model."""
#     import os
#     import pickle
#     import numpy as np

#     # Load the model
#     model_path = model_artifact.path + ".pkl"
#     print(f"Loading model from: {model_path}")

#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         print("✅ Model loaded successfully")
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")
#         raise

#     # Extract model components
#     similarity_matrix = model['similarity_matrix']
#     total_wines = similarity_matrix.shape[0]
#     print(f"Evaluating model with {total_wines} wines...")

#     # Calculate coverage (percentage of wines with strong recommendations)
#     coverage = 0

#     for i in range(total_wines):
#         # Get top 5 recommendations for each wine (excluding itself)
#         sim_scores = [(j, similarity_matrix[i, j])
#                       for j in range(total_wines) if j != i]
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]

#         # If we have at least one recommendation with similarity > 0.5, count it as covered
#         if any(score > 0.5 for _, score in sim_scores):
#             coverage += 1

#     coverage_score = coverage / total_wines
#     print(f"✅ Evaluation completed. Quality score: {coverage_score:.2f}")
#     return coverage_score


# @component(packages_to_install=["google-cloud-storage"])
# def upload_model(model_artifact: Input[Model], uploaded_model_artifact: Output[Model]):
#     """Uploads the wine recommendation model artifact."""
#     import os
#     import shutil

#     # Create a directory structure for the model
#     model_dir = os.path.dirname(uploaded_model_artifact.path)
#     os.makedirs(model_dir, exist_ok=True)

#     # Define the path where the model file will be saved
#     model_file_path = os.path.join(model_dir, "model.pkl")
#     source_path = model_artifact.path + ".pkl"

#     # Copy the model file
#     print(f"Copying model from {source_path} to {model_file_path}")
#     shutil.copy(source_path, model_file_path)

#     # Set the URI to the directory containing the model
#     uploaded_model_artifact.uri = model_dir
#     print(f"✅ Model uploaded to directory: {model_dir}")
#     print(f"✅ Model URI set to: {uploaded_model_artifact.uri}")


# @component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
#            base_image="python:3.9")
# def register_model(
#     model_artifact: Input[Model],
#     registered_model: Output[Model],
#     model_display_name: str,
#     serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2",
# ):
#     """Registers the wine recommendation model to Vertex AI Model Registry."""
#     from google.cloud import aiplatform

#     aiplatform.init(project=project, location=region)

#     try:
#         model = aiplatform.Model.upload(
#             display_name=model_display_name,
#             artifact_uri=model_artifact.uri,
#             serving_container_image_uri=serving_container_image_uri,
#         )
#         print(f"✅ Model registered: {model.resource_name}")

#         # Store model information
#         registered_model.uri = model.resource_name
#         registered_model.metadata["display_name"] = model_display_name
#         registered_model.metadata["resource_name"] = model.resource_name

#     except Exception as e:
#         print(f"❌ Error registering model: {e}")
#         raise e


# @component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
#            base_image="python:3.9")
# def run_wine_batch_prediction(
#     model_registry_name: Input[Model],
#     batch_prediction_job: Output[Artifact],
#     input_data_uri: str,
#     output_uri_prefix: str,
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2",
#     machine_type: str = "n1-standard-4"
# ):
#     """Runs a batch prediction job for the wine recommendation model."""
#     from google.cloud import aiplatform
#     import time

#     # Initialize Vertex AI
#     aiplatform.init(project=project, location=region)

#     # Get the model resource name
#     model_name = model_registry_name.metadata.get("resource_name")
#     if not model_name:
#         print(f"❌ Error: Model resource_name not found in metadata.")
#         raise ValueError("Model resource_name is missing from metadata.")

#     try:
#         model = aiplatform.Model(model_name)
#         print(f"✅ Retrieved model: {model.resource_name}")
#     except Exception as e:
#         print(f"❌ Error retrieving model: {e}")
#         raise e

#     batch_job_display_name = f"wine-rec-batch-{int(time.time())}"

#     try:
#         # Launch batch prediction job
#         print(f"Starting batch prediction job...")

#         batch_prediction_job_obj = model.batch_predict(
#             job_display_name=batch_job_display_name,
#             gcs_source=input_data_uri,
#             gcs_destination_prefix=output_uri_prefix,
#             machine_type=machine_type,
#             starting_replica_count=1,
#             max_replica_count=1,
#             model_parameters={"content_type": "application/jsonl"}
#         )

#         print(
#             f"✅ Batch prediction job started: {batch_prediction_job_obj.resource_name}")

#         # Set metadata
#         batch_prediction_job.metadata = {
#             "resource_name": batch_prediction_job_obj.resource_name,
#             "display_name": batch_job_display_name,
#             "model_name": model.resource_name,
#             "output_location": output_uri_prefix
#         }

#         # Monitor the job
#         print(f"Monitoring batch prediction job...")
#         batch_prediction_job_obj.wait()

#         # Check final status
#         print(
#             f"Batch prediction job completed with state: {batch_prediction_job_obj.state}")

#         if batch_prediction_job_obj.state == aiplatform.compat.types.job_state.JobState.JOB_STATE_SUCCEEDED:
#             print(f"✅ Batch prediction completed successfully!")
#             print(f"Output data available at: {output_uri_prefix}")
#         else:
#             print(f"❌ Batch prediction job failed or was cancelled.")
#             print(f"Final state: {batch_prediction_job_obj.state}")

#     except Exception as e:
#         print(f"❌ Batch prediction error: {e}")
#         raise e


# @component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
#            base_image="python:3.9")
# def deploy_to_vertex_endpoint(
#     model_registry_name: Input[Model],
#     endpoint: Output[Model],
#     endpoint_display_name: str,
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2"
# ):
#     """Deploys a model to Vertex AI Endpoint."""
#     from google.cloud import aiplatform

#     # Initialize Vertex AI
#     aiplatform.init(project=project, location=region)

#     # Get the model resource name
#     model_name = model_registry_name.metadata.get("resource_name")
#     if not model_name:
#         print(f"❌ Error: Model resource_name not found in metadata.")
#         raise ValueError("Model resource_name is missing from metadata.")

#     try:
#         model = aiplatform.Model(model_name)
#         print(f"✅ Retrieved model: {model.resource_name}")
#     except Exception as e:
#         print(f"❌ Error retrieving model: {e}")
#         raise e

#     deployed_model_display_name = f"{model_registry_name.metadata.get('display_name', 'model')}-deployed"

#     try:
#         # Check if endpoint already exists
#         endpoints = aiplatform.Endpoint.list(
#             filter=f'display_name="{endpoint_display_name}"',
#             order_by="create_time desc",
#             project=project,
#             location=region,
#         )

#         if endpoints:
#             endpoint_to_use = endpoints[0]
#             print(
#                 f"⚠️ Using existing endpoint: {endpoint_to_use.resource_name}")
#         else:
#             endpoint_to_use = aiplatform.Endpoint.create(
#                 display_name=endpoint_display_name,
#                 project=project,
#                 location=region
#             )
#             print(f"✅ Created new endpoint: {endpoint_to_use.resource_name}")

#         # Deploy the model
#         print(f"Starting deployment of model to endpoint...")
#         endpoint_to_use.deploy(
#             model=model,
#             deployed_model_display_name=deployed_model_display_name,
#             machine_type="n1-standard-2",
#             min_replica_count=1,
#             max_replica_count=1,
#             traffic_split={"0": 100}
#         )

#         print(f"✅ Model deployed to endpoint: {endpoint_to_use.resource_name}")
#         endpoint.uri = endpoint_to_use.resource_name

#     except Exception as e:
#         print(f"❌ Deployment error: {e}")
#         # If endpoint was created, still save its URI
#         if 'endpoint_to_use' in locals():
#             endpoint.uri = endpoint_to_use.resource_name
#         raise e


# @dsl.pipeline(
#     name="wine-recommendation-pipeline",
#     description="End-to-end pipeline for training, evaluating and deploying a wine recommendation model."
# )
# def wine_pipeline(
#     data_path: str,
#     model_display_name: str = "wine-recommendation-model",
#     endpoint_display_name: str = "wine-recommendation-endpoint",
#     project: str = "dev2-ea8f",
#     region: str = "europe-west2",
#     evaluation_threshold: float = 0.6
# ):
#     # Load the wine data
#     load_task = load_data(data_path=data_path)

#     # Preprocess the data
#     preprocess_task = preprocess_data(
#         input_data=load_task.outputs["output_data"]
#     )

#     # Train the model
#     train_task = train_model(
#         input_data=preprocess_task.outputs["output_data"]
#     )

#     # Evaluate the model
#     evaluate_task = evaluate_model(
#         model_artifact=train_task.outputs["output_model"]
#     )

#     # Use with_parameter to create a conditional branch
#     with dsl.Condition(
#         evaluate_task.output >= evaluation_threshold,
#         name="check_model_quality"
#     ):
#         # Upload the model only if evaluation passes threshold
#         upload_task = upload_model(
#             model_artifact=train_task.outputs["output_model"]
#         )

#         # Register the model in Vertex AI Model Registry
#         register_task = register_model(
#             model_artifact=upload_task.outputs["uploaded_model_artifact"],
#             model_display_name=model_display_name,
#             project=project,
#             region=region
#         )

#         # Run batch prediction
#         # batch_predict_task = run_wine_batch_prediction(
#         #     model_registry_name=register_task.outputs["registered_model"],
#         #     input_data_uri=f"gs://model-output-wine-dev2-ea8f/batch-inputs/wine_input.jsonl",
#         #     output_uri_prefix=f"gs://model-output-wine-dev2-ea8f/batch-outputs/",
#         #     project=project,
#         #     region=region
#         # )

#         # Deploy the model to a Vertex AI endpoint
#         # deploy_task = deploy_to_vertex_endpoint(
#         #     model_registry_name=register_task.outputs["registered_model"],
#         #     endpoint_display_name=endpoint_display_name,
#         #     project=project,
#         #     region=region
#         # ).after(batch_predict_task)
#         deploy_task = deploy_to_vertex_endpoint(
#             model_registry_name=register_task.outputs["registered_model"],
#             endpoint_display_name=endpoint_display_name,
#             project=project,
#             region=region
#         )


# # Compile the pipeline
# def compile_pipeline(output_file="wine_recommendation_pipeline.json"):
#     compiler.Compiler().compile(
#         pipeline_func=wine_pipeline,
#         package_path=output_file
#     )


# # Function to run the pipeline
# def run_pipeline(
#     project_id,
#     gcs_bucket,
#     data_path,
#     model_display_name="wine-recommendation-model",
#     endpoint_display_name="wine-recommendation-endpoint",
#     region="europe-west2",
#     pipeline_root=None,
#     evaluation_threshold=0.5
# ):
#     """Run the wine recommendation pipeline on Vertex AI."""
#     from google.cloud import aiplatform

#     # Initialize Vertex AI
#     aiplatform.init(project=project_id, location=region)

#     # Set pipeline root if not provided
#     if pipeline_root is None:
#         pipeline_root = f"gs://{gcs_bucket}/pipeline_root/wine_recommendation"

#     # Compile the pipeline
#     pipeline_file = "wine_recommendation_pipeline.json"
#     compile_pipeline(pipeline_file)

#     # Create a pipeline job
#     job = aiplatform.PipelineJob(
#         display_name="wine-recommendation-job",
#         template_path=pipeline_file,
#         pipeline_root=pipeline_root,
#         parameter_values={
#             "data_path": data_path,
#             "model_display_name": model_display_name,
#             "endpoint_display_name": endpoint_display_name,
#             "project": project_id,
#             "region": region,
#             "evaluation_threshold": evaluation_threshold
#         },
#         enable_caching=True
#     )

#     # Run the pipeline
#     job.run(sync=True)

#     return job
