# rating_prediction_pipeline.py
from kfp.v2 import dsl
from kfp.v2.dsl import Dataset, Model, Input, Output, component
from kfp.v2 import compiler
from rating_prediction_constants import BASE_IMAGE, GCS_BUCKET  

# 1. Load data
@component(packages_to_install=["pandas", "google-cloud-storage"], base_image=BASE_IMAGE)
def load_data(data_path: str, output_data: Output[Dataset]):
    import pandas as pd
    from google.cloud import storage
    import io
    
    bucket_name = data_path.replace('gs://', '').split('/')[0]
    blob_name = '/'.join(data_path.replace('gs://', '').split('/')[1:])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_data))
    
    df.to_csv(output_data.path, index=False)
    print(f"Loaded {len(df)} records from GCS")


# 2. Train model
@component(packages_to_install=["pandas==1.3.5", "scikit-learn==0.24.2", "numpy==1.21.6"], base_image=BASE_IMAGE)
def train_model(input_data: Input[Dataset], model_output: Output[Model]):
    import pandas as pd
    import pickle
    import re
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    df = pd.read_csv(input_data.path)
    print(f"Loaded {len(df)} records")
    print(f"Available columns: {list(df.columns)}")
    
    # categorical columns
    categorical_cols = ['Country', 'Type', 'Grape', 'Style', 'Region']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # extract price
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 10.0
        )
        print("INFO: Extracted price_numeric from Price column")
    else:
        df["price_numeric"] = 10.0
        print("WARNING: No Price column, using default 10.0")
    
    # rating target from price
    df['Rating'] = (df['price_numeric'] * 0.15) + 3.0
    df.loc[df['Rating'] > 5.0, 'Rating'] = 5.0
    df.loc[df['Rating'] < 3.0, 'Rating'] = 3.0
    print("INFO: Created Rating target from price")
    
    features = ['price_numeric']
    categorical_features = []
    
    # Add categorical features
    for col in ['Country', 'Type', 'Grape', 'Style']:
        if col in df.columns:
            features.append(col)
            categorical_features.append(col)
    
    print(f"Using features: {features}")
    print(f"Categorical features: {categorical_features}")
    
    transformers = []
    transformers.append(('num', 'passthrough', [0]))
    if categorical_features:
        cat_indices = list(range(1, len(features)))
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_indices))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    
    # Train
    X = df[features].values
    y = df['Rating'].values
    
    print(f"Training data shape: {X.shape}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    
    pipeline.fit(X, y)
    
    # Save model
    with open(model_output.path + ".pkl", 'wb') as f:
        pickle.dump(pipeline, f)
    
    model_output.metadata["features"] = str(features)
    print("INFO: Model trained successfully!")


# # 3. Register model
# @component(packages_to_install=["google-cloud-aiplatform"], base_image=BASE_IMAGE)
# def register_model(model_input: Input[Model], model_output: Output[Model], 
#                    model_name: str, project: str, region: str):
#     from google.cloud import aiplatform
#     import shutil
#     import os
    
#     model_dir = os.path.dirname(model_output.path)
#     os.makedirs(model_dir, exist_ok=True)
#     source_file = model_input.path + ".pkl"
#     target_file = os.path.join(model_dir, "model.pkl")
    
#     print(f"INFO: Copying model from: {source_file}")
#     print(f"INFO: Copying model to: {target_file}")
    
#     if os.path.exists(source_file):
#         shutil.copy(source_file, target_file)
#         print("INFO: Model file copied successfully")
#     else:
#         raise FileNotFoundError(f"Source model file not found: {source_file}")
    
#     if os.path.exists(target_file):
#         print(f"INFO: Model file verified at: {target_file}")
#         print(f"INFO: File size: {os.path.getsize(target_file)} bytes")
#     else:
#         raise FileNotFoundError(f"Target model file not found: {target_file}")
    
#     aiplatform.init(project=project, location=region)
    
#     print(f"Registering model from directory: {model_dir}")
#     print(f"Directory contents: {os.listdir(model_dir)}")
    
#     model = aiplatform.Model.upload(
#         display_name=model_name,
#         artifact_uri=model_dir,
#         serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
#     )
    
#     model_output.uri = model_dir
#     model_output.metadata["resource_name"] = model.resource_name
#     model_output.metadata["display_name"] = model_name
    
#     if hasattr(model_input, 'metadata'):
#         for key, value in model_input.metadata.items():
#             if key not in model_output.metadata:
#                 model_output.metadata[key] = value
    
#     print(f"INFO: Model registered: {model.resource_name}")

# 3. Register model
@component(packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"], base_image=BASE_IMAGE)
def register_model(model_input: Input[Model], model_output: Output[Model], 
                   model_name: str, project: str, region: str, gcs_bucket: str):
    from google.cloud import aiplatform, storage
    import os
    
    # Upload model directly to GCS
    source_file = model_input.path + ".pkl"
    gcs_model_path = f"gs://{gcs_bucket}/models/{model_name}"
    
    print(f"INFO: Uploading model to GCS: {gcs_model_path}")
    print(f"INFO: Source file: {source_file}")
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(f"models/{model_name}/model.pkl")
    blob.upload_from_filename(source_file)
    
    print(f"INFO: Model uploaded to GCS successfully")
    
    # Register to Vertex AI
    aiplatform.init(project=project, location=region)
    
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=gcs_model_path,
        serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    )
    
    # Set outputs
    model_output.uri = gcs_model_path
    model_output.metadata["resource_name"] = model.resource_name
    model_output.metadata["display_name"] = model_name
    
    if hasattr(model_input, 'metadata'):
        for key, value in model_input.metadata.items():
            if key not in model_output.metadata:
                model_output.metadata[key] = value
    
    print(f"INFO: Model registered: {model.resource_name}")


# 4. Batch predict
@component(packages_to_install=["google-cloud-aiplatform"], base_image=BASE_IMAGE)
def batch_predict(model_input: Input[Model], 
                  input_path: str, output_path: str, 
                  project: str, region: str,
                  machine_type: str = "n1-standard-4"):
    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=region)
    model_name = model_input.metadata["resource_name"]
    model = aiplatform.Model(model_name)
    
    print(f"INFO: Using model: {model_name}")
    print(f"INFO: Input data: {input_path}")
    print(f"INFO: Output path: {output_path}")
    print(f"INFO: Machine type: {machine_type}")
    
    job = model.batch_predict(
        job_display_name="rating-prediction-batch-job",
        gcs_source=input_path,
        gcs_destination_prefix=output_path,
        machine_type=machine_type,
        starting_replica_count=1,
        max_replica_count=1,
        sync=True
    )
    
    print(f"INFO: Batch prediction completed: {job.resource_name}")
    print(f"INFO: Check results at: {output_path}")


# Pipeline definition
@dsl.pipeline(name="rating-prediction-batch-pipeline")
def rating_prediction_pipeline(
    data_path: str,
    batch_input_path: str,
    batch_output_path: str,
    project: str,
    region: str,
    machine_type: str ,
    model_name: str,
):
    # Step 1: Load data
    load_task = load_data(data_path=data_path)
    
    # Step 2: Train model  
    train_task = train_model(input_data=load_task.outputs["output_data"])
    
    # Step 3: Register model
    register_task = register_model(
        model_input=train_task.outputs["model_output"],
        model_name=model_name,
        project=project,
        region=region,
        gcs_bucket=GCS_BUCKET
    )
    
    # Step 4: Batch predict
    batch_task = batch_predict(
        model_input=register_task.outputs["model_output"],
        input_path=batch_input_path,
        output_path=batch_output_path,
        project=project,
        region=region,
        machine_type=machine_type
    )


# Compile function
def compile_pipeline(output_file: str = "rating_prediction_pipeline.json"):
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=rating_prediction_pipeline,
        package_path=output_file
    )
    print(f"INFO: Pipeline compiled to {output_file}")


if __name__ == "__main__":
    compile_pipeline()