# super_simple_wine_pipeline.py
from kfp.v2 import dsl
from kfp.v2.dsl import Dataset, Model, Input, Output, component
from kfp.v2 import compiler


# 1. Load data
@component(packages_to_install=["pandas", "google-cloud-storage"], base_image="python:3.9")
def load_data(data_path: str, output_data: Output[Dataset]):
    import pandas as pd
    from google.cloud import storage
    import io
    
    # Parse GCS path
    bucket_name = data_path.replace('gs://', '').split('/')[0]
    blob_name = '/'.join(data_path.replace('gs://', '').split('/')[1:])
    
    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Read CSV data
    csv_data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_data))
    
    df.to_csv(output_data.path, index=False)
    print(f"Loaded {len(df)} wines from GCS")


# 2. Train model
@component(packages_to_install=["pandas", "scikit-learn"], base_image="python:3.9")
def train_model(input_data: Input[Dataset], model_output: Output[Model]):
    import pandas as pd
    import pickle
    import re
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    df = pd.read_csv(input_data.path)
    print(f"Loaded {len(df)} wines")
    print(f"Available columns: {list(df.columns)}")
    
    # Clean categorical columns
    categorical_cols = ['Country', 'Type', 'Grape', 'Style', 'Region']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Extract price_numeric from Price column
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 10.0
        )
        print("✅ Extracted price_numeric from Price column")
    else:
        df["price_numeric"] = 10.0
        print("⚠️ No Price column, using default 10.0")
    
    # Create Rating target from price
    df['Rating'] = (df['price_numeric'] * 0.15) + 3.0  # Scale price to 3-5 rating
    df.loc[df['Rating'] > 5.0, 'Rating'] = 5.0
    df.loc[df['Rating'] < 3.0, 'Rating'] = 3.0
    print("✅ Created Rating target from price")
    
    # Use simple features that exist in your data
    features = ['price_numeric']
    categorical_features = []
    
    # Add categorical features that exist
    for col in ['Country', 'Type', 'Grape', 'Style']:
        if col in df.columns:
            features.append(col)
            categorical_features.append(col)
    
    print(f"Using features: {features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create preprocessor
    transformers = []
    
    # Numeric feature (price)
    transformers.append(('num', 'passthrough', [0]))  # price_numeric is first
    
    # Categorical features
    if categorical_features:
        cat_indices = list(range(1, len(features)))  # all except first (price)
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_indices))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Create pipeline
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
    print("✅ Model trained successfully!")


# 3. Register model
@component(packages_to_install=["google-cloud-aiplatform"], base_image="python:3.9")
def register_model(model_input: Input[Model], model_output: Output[Model], 
                   model_name: str, project: str, region: str):
    from google.cloud import aiplatform
    import shutil
    import os
    
    # Create proper directory structure
    model_dir = os.path.dirname(model_output.path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy model with correct name
    source_file = model_input.path + ".pkl"
    target_file = os.path.join(model_dir, "model.pkl")
    
    print(f"Copying model from: {source_file}")
    print(f"Copying model to: {target_file}")
    
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print("✅ Model file copied successfully")
    else:
        raise FileNotFoundError(f"Source model file not found: {source_file}")
    
    # Verify the file exists
    if os.path.exists(target_file):
        print(f"✅ Model file verified at: {target_file}")
        print(f"✅ File size: {os.path.getsize(target_file)} bytes")
    else:
        raise FileNotFoundError(f"Target model file not found: {target_file}")
    
    # Register to Vertex AI
    aiplatform.init(project=project, location=region)
    
    print(f"Registering model from directory: {model_dir}")
    print(f"Directory contents: {os.listdir(model_dir)}")
    
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_dir,
        serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    
    # Set outputs
    model_output.uri = model_dir
    model_output.metadata["resource_name"] = model.resource_name
    model_output.metadata["display_name"] = model_name
    
    # Copy metadata from training
    if hasattr(model_input, 'metadata'):
        for key, value in model_input.metadata.items():
            if key not in model_output.metadata:
                model_output.metadata[key] = value
    
    print(f"✅ Model registered: {model.resource_name}")

# 4. Batch predict
@component(packages_to_install=["google-cloud-aiplatform"], base_image="python:3.9")
def batch_predict(model_input: Input[Model], 
                  input_path: str, output_path: str, 
                  project: str, region: str,
                  machine_type: str = "n1-standard-4"):
    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=region)
    model_name = model_input.metadata["resource_name"]
    model = aiplatform.Model(model_name)
    
    print(f"✅ Using model: {model_name}")
    print(f"✅ Input data: {input_path}")
    print(f"✅ Output path: {output_path}")
    print(f"✅ Machine type: {machine_type}")
    
    job = model.batch_predict(
        job_display_name="simple-batch-job",
        gcs_source=input_path,
        gcs_destination_prefix=output_path,
        machine_type=machine_type,
        starting_replica_count=1,
        max_replica_count=1,
        sync=True
    )
    
    print(f"✅ Batch prediction completed: {job.resource_name}")
    print(f"✅ Check results at: {output_path}")


# wine rating pipeline
@dsl.pipeline(name="wine-rating-batch-pipeline")
def wine_pipeline(
    data_path: str,
    batch_input_path: str,
    batch_output_path: str,
    project: str,
    region: str,
    serving_container_image_uri: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
):
    # Step 1: Load data
    load_task = load_data(data_path=data_path)
    
    # Step 2: Train model  
    train_task = train_model(input_data=load_task.outputs["output_data"])
    
    # Step 3: Register model
    register_task = register_model(
        model_input=train_task.outputs["model_output"],
        model_name="wine-rating-batch-model",
        project=project,
        region=region
    )
    
    # Step 4: Batch predict
    batch_task = batch_predict(
        model_input=register_task.outputs["model_output"],
        input_path=batch_input_path,
        output_path=batch_output_path,
        project=project,
        region=region
    )


# Compile
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=wine_pipeline,
        package_path="wine_pipeline.json"
    )
    print("Pipeline compiled!")