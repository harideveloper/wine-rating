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


# 3. Save model to GCS (no registry)
@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def save_model_to_gcs(model_input: Input[Model], gcs_model_path: str) -> str:
    """Save model directly to GCS without registry"""
    from google.cloud import storage
    import os
    
    source_file = model_input.path + ".pkl"
    
    # Parse GCS path
    path_parts = gcs_model_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1]
    
    print(f"INFO: Uploading model to GCS: {gcs_model_path}")
    print(f"INFO: Source file: {source_file}")
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Model file not found: {source_file}")
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file)
    
    print(f"INFO: Model uploaded to GCS successfully")
    print(f"INFO: Model size: {os.path.getsize(source_file)} bytes")
    
    return gcs_model_path


# 4. Direct batch prediction (no container dependency)
@component(packages_to_install=[
    "pandas==1.3.5", 
    "scikit-learn==0.24.2", 
    "numpy==1.21.6", 
    "google-cloud-storage"
], base_image=BASE_IMAGE)
def direct_batch_predict(
    gcs_model_path: str,
    input_path: str, 
    output_path: str
) -> str:
    """Direct batch prediction without Vertex AI containers"""
    import pandas as pd
    import pickle
    import json
    import re
    from google.cloud import storage
    import io
    import os
    
    client = storage.Client()
    
    # Download model from GCS
    print(f"INFO: Downloading model from: {gcs_model_path}")
    model_parts = gcs_model_path.replace("gs://", "").split("/", 1)
    model_bucket = client.bucket(model_parts[0])
    model_blob = model_bucket.blob(model_parts[1])
    model_blob.download_to_filename("model.pkl")
    
    # Load model
    with open("model.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    print("INFO: Model loaded successfully")
    
    # Download input data
    print(f"INFO: Processing input data from: {input_path}")
    input_parts = input_path.replace("gs://", "").split("/", 1)
    input_bucket = client.bucket(input_parts[0])
    input_blob = input_bucket.blob(input_parts[1])
    
    # Handle different input formats
    if input_path.endswith('.csv'):
        csv_data = input_blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
        print(f"INFO: Loaded {len(df)} records from CSV")
    elif input_path.endswith('.jsonl'):
        jsonl_data = input_blob.download_as_text()
        records = []
        for line in jsonl_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                if "instances" in data:
                    records.extend(data["instances"])
                else:
                    records.append(data)
        df = pd.DataFrame(records)
        print(f"INFO: Loaded {len(df)} records from JSONL")
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    print(f"INFO: Input data columns: {list(df.columns)}")
    
    # Prepare features (same logic as training)
    categorical_cols = ['Country', 'Type', 'Grape', 'Style', 'Region']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Extract price if available
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(
            lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
            if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
            else 10.0
        )
    elif "price_numeric" not in df.columns:
        df["price_numeric"] = 10.0
        print("WARNING: No Price or price_numeric column, using default 10.0")
    
    # Prepare feature matrix
    features = ['price_numeric']
    for col in ['Country', 'Type', 'Grape', 'Style']:
        if col in df.columns:
            features.append(col)
    
    print(f"INFO: Using features for prediction: {features}")
    
    # Make predictions
    X = df[features].values
    predictions = model_pipeline.predict(X)
    
    print(f"INFO: Generated {len(predictions)} predictions")
    print(f"INFO: Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
    
    # Prepare output
    results = []
    for i, pred in enumerate(predictions):
        result = {
            "prediction": float(pred),
            "input_index": i
        }
        # Add input features to result for reference
        for j, feature in enumerate(features):
            if j < len(X[i]):
                result[f"input_{feature}"] = str(X[i][j]) if isinstance(X[i][j], (str, object)) else float(X[i][j])
        results.append(result)
    
    # Save results to GCS
    output_parts = output_path.replace("gs://", "").split("/", 1)
    output_bucket = client.bucket(output_parts[0])
    
    # Save as JSONL
    output_file = "predictions.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Upload to GCS
    output_blob_name = f"{output_parts[1]}/predictions.jsonl" if len(output_parts) > 1 else "predictions.jsonl"
    output_blob = output_bucket.blob(output_blob_name)
    output_blob.upload_from_filename(output_file)
    
    final_output_path = f"gs://{output_parts[0]}/{output_blob_name}"
    print(f"INFO: Predictions saved to: {final_output_path}")
    
    return f"Batch prediction completed. {len(predictions)} predictions saved to {final_output_path}"


# 5. Validate results
@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def validate_results(output_path: str) -> str:
    """Validate batch prediction results"""
    from google.cloud import storage
    import json
    
    client = storage.Client()
    
    # Find prediction files
    if output_path.startswith("gs://"):
        path_parts = output_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        prefix = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
    else:
        raise ValueError(f"Invalid GCS path: {output_path}")
    
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    prediction_files = [blob.name for blob in blobs if blob.name.endswith('.jsonl')]
    
    if not prediction_files:
        return f"ERROR: No prediction files found at {output_path}"
    
    # Validate first prediction file
    prediction_blob = bucket.blob(prediction_files[0])
    prediction_data = prediction_blob.download_as_text()
    
    prediction_count = 0
    sample_predictions = []
    
    for line in prediction_data.strip().split('\n'):
        if line.strip():
            prediction = json.loads(line)
            prediction_count += 1
            if len(sample_predictions) < 3:
                sample_predictions.append(prediction)
    
    print(f"INFO: Found {len(prediction_files)} prediction files")
    print(f"INFO: Total predictions: {prediction_count}")
    print(f"INFO: Sample predictions: {sample_predictions}")
    
    return f"Validation successful. {prediction_count} predictions in {len(prediction_files)} files."


# Pipeline definition
@dsl.pipeline(name="rating-prediction-batch-pipeline-direct")
def rating_prediction_pipeline(
    data_path: str,
    batch_input_path: str,
    batch_output_path: str,
    model_name: str,
    gcs_bucket: str = GCS_BUCKET
):
    """Rating prediction pipeline with direct batch prediction (no registry)"""
    
    # Step 1: Load data
    load_task = load_data(data_path=data_path)
    
    # Step 2: Train model  
    train_task = train_model(input_data=load_task.outputs["output_data"])
    
    # Step 3: Save model to GCS (no registry)
    gcs_model_path = f"gs://{gcs_bucket}/models/{model_name}/model.pkl"
    save_task = save_model_to_gcs(
        model_input=train_task.outputs["model_output"],
        gcs_model_path=gcs_model_path
    )
    
    # Step 4: Direct batch prediction
    predict_task = direct_batch_predict(
        gcs_model_path=save_task.output,
        input_path=batch_input_path,
        output_path=batch_output_path
    )
    
    # Step 5: Validate results
    validate_task = validate_results(
        output_path=batch_output_path
    )
    
    # Set dependencies
    train_task.after(load_task)
    save_task.after(train_task)
    predict_task.after(save_task)
    validate_task.after(predict_task)


# Compile function
def compile_pipeline(output_file: str = "rating_prediction_pipeline_direct.json"):
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=rating_prediction_pipeline,
        package_path=output_file
    )
    print(f"INFO: Pipeline compiled to {output_file}")


if __name__ == "__main__":
    compile_pipeline()