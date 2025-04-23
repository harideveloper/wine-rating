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
    """Preprocesses the wine data for the prediction model."""
    import os
    import pandas as pd
    import re

    # Load the input data
    df = pd.read_csv(input_data.path)
    print(f"✅ Loaded input data with shape: {df.shape}")

    # Handle missing values in categorical columns
    categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Extract numeric price values if available
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 0
        )
    else:
        df["price_numeric"] = 0

    # Create target column if needed
    if 'Rating' not in df.columns:
        # Create a synthetic rating based on price (just for demonstration purposes)
        # In a real scenario, you might have actual ratings or use a different approach
        df['Rating'] = df['price_numeric'] * 0.2 + 3.0
        df.loc[df['Rating'] > 5.0, 'Rating'] = 5.0
        print("⚠️ Created synthetic Rating column based on price")

    # Split data into train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save train/test data
    train_path = os.path.join(os.path.dirname(
        output_data.path), "train_data.csv")
    test_path = os.path.join(os.path.dirname(
        output_data.path), "test_data.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save combined data for the pipeline output
    df.to_csv(output_data.path, index=False)

    print(
        f"✅ Preprocessed train data saved at {train_path} with shape {train_df.shape}")
    print(
        f"✅ Preprocessed test data saved at {test_path} with shape {test_df.shape}")
    print(f"✅ Combined preprocessed data saved at {output_data.path}")

    # Add paths as metadata
    output_data.metadata["train_path"] = train_path
    output_data.metadata["test_path"] = test_path

    return {
        "train_data": train_path,
        "test_data": test_path
    }


@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.0.1"],
           base_image="python:3.9")
def train_model(
    train_data_path: str,
    output_model: Output[Model]
):
    """Trains a wine rating prediction model using RandomForestRegressor."""
    import os
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Load the preprocessed training data
    df = pd.read_csv(train_data_path)
    print(f"✅ Loaded training data with shape: {df.shape}")

    # Define categorical and numeric features
    categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
    categorical_features = [
        col for col in categorical_features if col in df.columns]

    numeric_features = ['price_numeric']
    numeric_features = [col for col in numeric_features if col in df.columns]

    # Target is Rating
    target = 'Rating'

    # Create feature preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the model
    X = df[numeric_features + categorical_features]
    y = df[target]
    model_pipeline.fit(X, y)
    print("✅ Model trained successfully")

    # Save the model using pickle
    file_name = output_model.path + ".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model_pipeline, file)

    # Add metadata
    output_model.metadata["framework"] = "sklearn"
    output_model.metadata["features"] = str(
        numeric_features + categorical_features)
    output_model.metadata["target"] = target

    print(f"✅ Model trained and saved to {file_name}")


@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.0.1"])
def evaluate_model(
    model_artifact: Input[Model],
    test_data_path: str
) -> float:
    """Evaluates the wine rating prediction model."""
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Load the model
    model_path = model_artifact.path + ".pkl"
    print(f"Loading model from: {model_path}")

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    # Load test data
    test_df = pd.read_csv(test_data_path)
    print(f"✅ Loaded test data with shape: {test_df.shape}")

    # Get features from model metadata
    features_str = model_artifact.metadata.get("features", "")
    if features_str:
        import ast
        features = ast.literal_eval(features_str)
    else:
        # Fallback to default features
        categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
        categorical_features = [
            col for col in categorical_features if col in test_df.columns]
        numeric_features = ['price_numeric']
        numeric_features = [
            col for col in numeric_features if col in test_df.columns]
        features = numeric_features + categorical_features

    target = model_artifact.metadata.get("target", "Rating")

    # Make predictions
    X_test = test_df[features]
    y_test = test_df[target]

    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Evaluation Results:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")

    # Calculate a quality score (higher is better)
    # Normalize R² to be between 0 and 1 (R² can be negative for poor models)
    r2_norm = max(0, r2)

    # Convert RMSE to a 0-1 score (lower RMSE is better)
    # Assuming a maximum RMSE of 1.0 for a wine rating model
    rmse_score = max(0, 1.0 - rmse)

    # Combine the metrics with weights
    quality_score = (0.7 * r2_norm) + (0.3 * rmse_score)
    print(f"✅ Model quality score: {quality_score:.4f}")

    return quality_score


@component(packages_to_install=["google-cloud-storage"])
def upload_model(model_artifact: Input[Model], uploaded_model_artifact: Output[Model]):
    """Uploads the wine rating model artifact."""
    import os
    import shutil

    # Create a directory structure for the model
    model_dir = os.path.dirname(uploaded_model_artifact.path)
    os.makedirs(model_dir, exist_ok=True)

    # Define the path where the model file will be saved
    model_file_path = os.path.join(model_dir, "model.pkl")
    source_path = model_artifact.path + ".pkl"

    # Copy the model file
    print(f"Copying model from {source_path} to {model_file_path}")
    shutil.copy(source_path, model_file_path)

    # Set the URI to the directory containing the model
    uploaded_model_artifact.uri = model_dir
    uploaded_model_artifact.metadata.update(model_artifact.metadata)

    print(f"✅ Model uploaded to directory: {model_dir}")
    print(f"✅ Model URI set to: {uploaded_model_artifact.uri}")


@component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
           base_image="python:3.9")
def register_model(
    model_artifact: Input[Model],
    registered_model: Output[Model],
    model_display_name: str,
    serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
):
    """Registers the wine rating model to Vertex AI Model Registry."""
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
        # Pass through other metadata
        for key, value in model_artifact.metadata.items():
            if key not in registered_model.metadata:
                registered_model.metadata[key] = value

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise e


@component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
           base_image="python:3.9")
def run_batch_prediction(
    model_registry_name: Input[Model],
    batch_prediction_job: Output[Artifact],
    input_data_uri: str,
    output_uri_prefix: str,
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
    machine_type: str = "n1-standard-4"
):
    """Runs a batch prediction job for the wine rating model."""
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

    batch_job_display_name = f"wine-rating-batch-{int(time.time())}"

    try:
        # Launch batch prediction job
        print(f"Starting batch prediction job...")

        batch_prediction_job_obj = model.batch_predict(
            job_display_name=batch_job_display_name,
            gcs_source=input_data_uri,
            gcs_destination_prefix=output_uri_prefix,
            machine_type=machine_type,
            starting_replica_count=1,
            max_replica_count=1,
            model_parameters={"content_type": "csv"}
        )

        print(
            f"✅ Batch prediction job started: {batch_prediction_job_obj.resource_name}")

        # Set metadata
        batch_prediction_job.metadata = {
            "resource_name": batch_prediction_job_obj.resource_name,
            "display_name": batch_job_display_name,
            "model_name": model.resource_name,
            "output_location": output_uri_prefix
        }

        # Monitor the job
        print(f"Monitoring batch prediction job...")
        batch_prediction_job_obj.wait()

        # Check final status
        print(
            f"Batch prediction job completed with state: {batch_prediction_job_obj.state}")

        if batch_prediction_job_obj.state == aiplatform.compat.types.job_state.JobState.JOB_STATE_SUCCEEDED:
            print(f"✅ Batch prediction completed successfully!")
            print(f"Output data available at: {output_uri_prefix}")
        else:
            print(f"❌ Batch prediction job failed or was cancelled.")
            print(f"Final state: {batch_prediction_job_obj.state}")

    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        raise e


@component(packages_to_install=["google-cloud-aiplatform>=1.22.1"],
           base_image="python:3.9")
def deploy_to_vertex_endpoint(
    model_registry_name: Input[Model],
    endpoint: Output[Model],
    endpoint_display_name: str,
    project: str = "dev2-ea8f",
    region: str = "europe-west2"
):
    """Deploys a model to Vertex AI Endpoint."""
    from google.cloud import aiplatform

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

        # Deploy the model
        print(f"Starting deployment of model to endpoint...")
        endpoint_to_use.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=1,
            traffic_split={"0": 100}
        )

        print(f"✅ Model deployed to endpoint: {endpoint_to_use.resource_name}")
        endpoint.uri = endpoint_to_use.resource_name

    except Exception as e:
        print(f"❌ Deployment error: {e}")
        # If endpoint was created, still save its URI
        if 'endpoint_to_use' in locals():
            endpoint.uri = endpoint_to_use.resource_name
        raise e


@dsl.pipeline(
    name="wine-rating-prediction-pipeline",
    description="End-to-end pipeline for training, evaluating and deploying a wine rating prediction model."
)
def wine_pipeline(
    data_path: str,
    model_display_name: str = "wine-rating-model",
    endpoint_display_name: str = "wine-rating-endpoint",
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
    evaluation_threshold: float = 0.6,
    batch_input_uri: str = "gs://model-output-wine-dev2-ea8f/batch-inputs/wine_input.csv",
    batch_output_uri: str = "gs://model-output-wine-dev2-ea8f/batch-outputs/"
):
    # Load the wine data
    load_task = load_data(data_path=data_path)

    # Preprocess the data
    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"]
    )

    # Train the model
    train_task = train_model(
        train_data_path=preprocess_task.outputs["train_data"]
    )

    # Evaluate the model
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["output_model"],
        test_data_path=preprocess_task.outputs["test_data"]
    )

    # Use dsl.Condition to create a conditional branch
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

        # Run batch prediction
        batch_predict_task = run_batch_prediction(
            model_registry_name=register_task.outputs["registered_model"],
            input_data_uri=batch_input_uri,
            output_uri_prefix=batch_output_uri,
            project=project,
            region=region
        )

        # Deploy the model to a Vertex AI endpoint
        deploy_task = deploy_to_vertex_endpoint(
            model_registry_name=register_task.outputs["registered_model"],
            endpoint_display_name=endpoint_display_name,
            project=project,
            region=region
        ).after(batch_predict_task)


# Compile the pipeline
def compile_pipeline(output_file="wine_rating_pipeline.json"):
    compiler.Compiler().compile(
        pipeline_func=wine_pipeline,
        package_path=output_file
    )


# Function to run the pipeline
def run_pipeline(
    project_id,
    gcs_bucket,
    data_path,
    model_display_name="wine-rating-model",
    endpoint_display_name="wine-rating-endpoint",
    region="europe-west2",
    pipeline_root=None,
    evaluation_threshold=0.6
):
    """Run the wine rating prediction pipeline on Vertex AI."""
    from google.cloud import aiplatform

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Set pipeline root if not provided
    if pipeline_root is None:
        pipeline_root = f"gs://{gcs_bucket}/pipeline_root/wine_rating"

    # Compile the pipeline
    pipeline_file = "wine_rating_pipeline.json"
    compile_pipeline(pipeline_file)

    # Create a pipeline job
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

    # Run the pipeline
    job.run(sync=True)

    return job
