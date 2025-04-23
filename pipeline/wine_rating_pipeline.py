# wine_rating_pipeline.py
from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Dataset, Model, Input, Output, component
from kfp.v2 import compiler
from typing import Dict, List


# load data
@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "gcsfs==2025.3.2"],
           base_image="python:3.9")
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

    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    df.to_csv(output_data.path, index=False)
    print(f"✅ Wine data saved to {output_data.path}")


# pre-process data
@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.0.1"],
           base_image="python:3.9")
def preprocess_data(input_data: Input[Dataset],
                    output_data: Output[Dataset],
                    train_data: Output[Dataset],
                    test_data: Output[Dataset]):
    """Preprocesses the wine data for the prediction model."""
    import os
    import pandas as pd
    import re
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    print(f"✅ Loaded input data with shape: {df.shape}")

    # Data cleanup ( handle missing columns & handle null price )
    categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 0
        )
    else:
        df["price_numeric"] = 0

    # target column (tating) for final prediction
    if 'Rating' not in df.columns:
        df['Rating'] = df['price_numeric'] * 0.2 + 3.0
        df.loc[df['Rating'] > 5.0, 'Rating'] = 5.0
        print("⚠️ Created synthetic Rating column based on price")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    df.to_csv(output_data.path, index=False)

    print(
        f"✅ Preprocessed train data saved at {train_data.path} with shape {train_df.shape}")
    print(
        f"✅ Preprocessed test data saved at {test_data.path} with shape {test_df.shape}")
    print(f"✅ Combined preprocessed data saved at {output_data.path}")


# train model
@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.0.1"],
           base_image="python:3.9")
def train_model(
    train_data: Input[Dataset],
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

    df = pd.read_csv(train_data.path)
    print(f"✅ Loaded training data with shape: {df.shape}")

    # features
    categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
    categorical_features = [
        col for col in categorical_features if col in df.columns]

    numeric_features = ['price_numeric']
    numeric_features = [col for col in numeric_features if col in df.columns]
    feature_order = numeric_features + categorical_features
    target = 'Rating'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [i for i, col in enumerate(
                feature_order) if col in numeric_features]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [
             i for i, col in enumerate(feature_order) if col in categorical_features])
        ],
        remainder='passthrough'
    )
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X = df[feature_order].values
    y = df[target].values
    model_pipeline.fit(X, y)
    print("✅ Model trained successfully")

    file_name = output_model.path + ".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model_pipeline, file)
    output_model.metadata["framework"] = "sklearn"
    output_model.metadata["feature_order"] = str(feature_order)
    output_model.metadata["target"] = target

    print(f"✅ Model trained and saved to {file_name}")


# evaluate model
@component(packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.0.1"],
           base_image="python:3.9")
def evaluate_model(
    model_artifact: Input[Model],
    test_data: Input[Dataset]
) -> float:
    """Evaluates the wine rating prediction model."""
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    model_path = model_artifact.path + ".pkl"
    print(f"Loading model from: {model_path}")

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    test_df = pd.read_csv(test_data.path)
    print(f"✅ Loaded test data with shape: {test_df.shape}")

    features_str = model_artifact.metadata.get("features", "")
    if features_str:
        import ast
        features = ast.literal_eval(features_str)
    else:
        categorical_features = ['Country', 'Region', 'Type', 'Style', 'Grape']
        categorical_features = [
            col for col in categorical_features if col in test_df.columns]
        numeric_features = ['price_numeric']
        numeric_features = [
            col for col in numeric_features if col in test_df.columns]
        features = numeric_features + categorical_features

    target = model_artifact.metadata.get("target", "Rating")

    # predictions
    X_test = test_df[features]
    y_test = test_df[target]

    y_pred = model.predict(X_test)

    # metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Evaluation Results:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")

    # Calculate a quality score (higher = better)
    r2_norm = max(0, r2)
    rmse_score = max(0, 1.0 - rmse)
    quality_score = (0.7 * r2_norm) + (0.3 * rmse_score)
    print(f"✅ Model quality score: {quality_score:.4f}")

    return quality_score


# upload model to registry
@component(packages_to_install=["google-cloud-storage"],
           base_image="python:3.9")
def upload_model(model_artifact: Input[Model], uploaded_model_artifact: Output[Model]):
    """Uploads the wine rating model artifact."""
    import os
    import shutil

    model_dir = os.path.dirname(uploaded_model_artifact.path)
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = os.path.join(model_dir, "model.pkl")
    source_path = model_artifact.path + ".pkl"
    print(f"Copying model from {source_path} to {model_file_path}")
    shutil.copy(source_path, model_file_path)
    uploaded_model_artifact.uri = model_dir
    uploaded_model_artifact.metadata.update(model_artifact.metadata)

    print(f"✅ Model uploaded to directory: {model_dir}")
    print(f"✅ Model URI set to: {uploaded_model_artifact.uri}")


# register model to model registry
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
        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name
        for key, value in model_artifact.metadata.items():
            if key not in registered_model.metadata:
                registered_model.metadata[key] = value

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise e


# deploy to endpoint
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

    aiplatform.init(project=project, location=region)
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
        if 'endpoint_to_use' in locals():
            endpoint.uri = endpoint_to_use.resource_name
        raise e


@dsl.pipeline(
    name="wine-rating-prediction-pipeline",
    description="End-to-end pipeline for training, evaluating and deploying a wine rating prediction model."
)
# wine pipeline
def wine_pipeline(
    data_path: str,
    model_display_name: str = "wine-rating-model",
    endpoint_display_name: str = "wine-rating-endpoint",
    project: str = "dev2-ea8f",
    region: str = "europe-west2",
    evaluation_threshold: float = 0.6,
):

    load_task = load_data(data_path=data_path)
    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"]
    )
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"]
    )
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["output_model"],
        test_data=preprocess_task.outputs["test_data"]
    )

    with dsl.Condition(
        evaluate_task.output >= evaluation_threshold,
        name="model deployment"
    ):
        upload_task = upload_model(
            model_artifact=train_task.outputs["output_model"]
        )
        register_task = register_model(
            model_artifact=upload_task.outputs["uploaded_model_artifact"],
            model_display_name=model_display_name,
            project=project,
            region=region
        )
        deploy_task = deploy_to_vertex_endpoint(
            model_registry_name=register_task.outputs["registered_model"],
            endpoint_display_name=endpoint_display_name,
            project=project,
            region=region
        )


# Compile the pipeline
def compile_pipeline(output_file: str = "wine_rating_pipeline.json"):
    compiler.Compiler().compile(
        pipeline_func=wine_pipeline,
        package_path=output_file
    )
