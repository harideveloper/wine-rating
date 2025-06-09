# rating_prediction_pipeline.py
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Dataset, Model, Input, Output, component
from rating_prediction_constants import BASE_IMAGE, GCS_BUCKET

# Constants for feature processing
CATEGORICAL_COLS = ['Country', 'Type', 'Grape', 'Style', 'Region']
FEATURE_COLS = ['price_numeric', 'Country', 'Type', 'Grape', 'Style']
RATING_BONUSES = {
    'Country': {'France': 0.3, 'Italy': 0.2, 'Spain': 0.1, 'Germany': 0.15, 'USA': 0.1, 'Australia': 0.05, 'Portugal': 0.1},
    'Type': {'Red': 0.1, 'White': 0.05, 'Rosé': 0.0, 'Sparkling': 0.15}
}

# Shared utility functions
def extract_price_numeric(price_series):
    """Extract numeric price from Price column."""
    import re
    import pandas as pd
    
    return price_series.apply(
        lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1))
        if pd.notna(x) and re.search(r'\d+\.?\d*', str(x))
        else 10.0
    )

def prepare_features(df):
    """Standardize feature preparation for both training and prediction."""
    import pandas as pd
    
    # Handle categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Handle price extraction
    if "Price" in df.columns:
        df["price_numeric"] = extract_price_numeric(df["Price"])
        print(f"INFO: Extracted price_numeric from Price column")
    elif "price_numeric" in df.columns:
        df["price_numeric"] = pd.to_numeric(df["price_numeric"], errors='coerce').fillna(10.0)
        print(f"INFO: Using existing price_numeric column")
    else:
        df["price_numeric"] = 10.0
        print("WARNING: No price data found, using default 10.0")
    
    # Select available features
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    print(f"INFO: Using features: {available_features}")
    
    return df, available_features

def load_gcs_data(data_path, file_format):
    """Load data from GCS in specified format."""
    from google.cloud import storage
    import pandas as pd
    import json
    import io
    
    # Parse GCS path
    bucket_name = data_path.replace('gs://', '').split('/')[0]
    blob_name = '/'.join(data_path.replace('gs://', '').split('/')[1:])
    
    # Download data
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if file_format == 'csv':
        csv_data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
    elif file_format == 'jsonl':
        jsonl_data = blob.download_as_text()
        records = []
        for line in jsonl_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                if "instances" in data:
                    records.extend(data["instances"])
                else:
                    records.append(data)
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    print(f"INFO: Loaded {len(df)} records from {file_format.upper()}")
    return df

def save_to_gcs(data, output_path, filename):
    """Save data to GCS."""
    from google.cloud import storage
    import json
    
    # Parse output path
    output_parts = output_path.replace("gs://", "").split("/", 1)
    bucket_name = output_parts[0]
    prefix = output_parts[1] if len(output_parts) > 1 else ""
    
    # Save locally first
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{prefix}/{filename}" if prefix else filename
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(filename)
    
    final_path = f"gs://{bucket_name}/{blob_name}"
    print(f"INFO: Data saved to: {final_path}")
    return final_path

# Pipeline Components
@component(packages_to_install=["pandas", "google-cloud-storage"], base_image=BASE_IMAGE)
def load_data(data_path: str, output_data: Output[Dataset]):
    """Load training data from GCS."""
    import pandas as pd
    
    df = load_gcs_data(data_path, 'csv')
    df.to_csv(output_data.path, index=False)

@component(packages_to_install=["pandas==1.3.5", "scikit-learn==0.24.2", "numpy==1.21.6"], base_image=BASE_IMAGE)
def train_model(input_data: Input[Dataset], model_output: Output[Model]):
    """Train the rating prediction model."""
    import pandas as pd
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Load and prepare data
    df = pd.read_csv(input_data.path)
    print(f"INFO: Training on {len(df)} records")
    
    df, features = prepare_features(df)
    
    # Create realistic rating targets
    base_rating = (df['price_numeric'] * 0.1) + 2.5
    
    # Add bonuses for variety
    for feature, bonus_map in RATING_BONUSES.items():
        if feature in df.columns:
            bonus = df[feature].map(bonus_map).fillna(0.0)
            base_rating += bonus
    
    # Add controlled noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(df))
    df['Rating'] = (base_rating + noise).clip(3.0, 5.0)
    
    print(f"INFO: Rating range: {df['Rating'].min():.2f} to {df['Rating'].max():.2f}")
    
    # Build pipeline
    categorical_features = [f for f in features if f in CATEGORICAL_COLS]
    transformers = [('num', 'passthrough', [0])]  # price_numeric at index 0
    
    if categorical_features:
        cat_indices = list(range(1, len(features)))
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_indices))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10))
    ])
    
    # Train model
    X = df[features]
    y = df['Rating'].values
    pipeline.fit(X, y)
    
    # Validate training
    test_pred = pipeline.predict(X.head())
    print(f"INFO: Model trained. Sample predictions: {test_pred[:3]}")
    
    # Save model
    with open(model_output.path + ".pkl", 'wb') as f:
        pickle.dump(pipeline, f)
    
    model_output.metadata["features"] = str(features)

@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def save_model_to_gcs(model_input: Input[Model], gcs_model_path: str) -> str:
    """Save trained model to GCS."""
    from google.cloud import storage
    import os
    
    source_file = model_input.path + ".pkl"
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Model file not found: {source_file}")
    
    # Parse GCS path and upload
    path_parts = gcs_model_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(path_parts[0])
    blob = bucket.blob(path_parts[1])
    blob.upload_from_filename(source_file)
    
    print(f"INFO: Model saved to {gcs_model_path} ({os.path.getsize(source_file)} bytes)")
    return gcs_model_path

@component(packages_to_install=[
    "pandas==1.3.5", "scikit-learn==0.24.2", "numpy==1.21.6", "google-cloud-storage"
], base_image=BASE_IMAGE)
def batch_predict(gcs_model_path: str, input_path: str, output_path: str) -> str:
    """Run batch predictions."""
    import pandas as pd
    import pickle
    from google.cloud import storage
    
    # Load model
    model_parts = gcs_model_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(model_parts[0])
    blob = bucket.blob(model_parts[1])
    blob.download_to_filename("model.pkl")
    
    with open("model.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    
    print("INFO: Model loaded successfully")
    
    # Load and prepare input data
    file_format = 'jsonl' if input_path.endswith('.jsonl') else 'csv'
    df = load_gcs_data(input_path, file_format)
    df, features = prepare_features(df)
    
    # Make predictions
    X = df[features]
    predictions = model_pipeline.predict(X)
    
    print(f"INFO: Generated {len(predictions)} predictions")
    print(f"INFO: Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
    
    # Prepare results
    results = []
    for i, pred in enumerate(predictions):
        result = {"prediction": float(pred), "input_index": i}
        for feature in features:
            result[f"input_{feature}"] = str(df.iloc[i][feature])
        results.append(result)
    
    # Save results
    final_path = save_to_gcs(results, output_path, "predictions.jsonl")
    return f"Batch prediction completed. {len(predictions)} predictions saved to {final_path}"

@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def validate_results(output_path: str) -> str:
    """Validate prediction results."""
    from google.cloud import storage
    import json
    
    # Find prediction files
    path_parts = output_path.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    prefix = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    prediction_files = [blob.name for blob in blobs if blob.name.endswith('.jsonl')]
    
    if not prediction_files:
        return f"ERROR: No prediction files found at {output_path}"
    
    # Sample validation
    prediction_blob = bucket.blob(prediction_files[0])
    prediction_data = prediction_blob.download_as_text()
    
    predictions = []
    for line in prediction_data.strip().split('\n'):
        if line.strip():
            predictions.append(json.loads(line))
    
    print(f"INFO: Found {len(predictions)} predictions")
    print(f"INFO: Sample: {predictions[:2]}")
    
    return f"Validation successful. {len(predictions)} predictions in {len(prediction_files)} files."

# Pipeline Definition
@dsl.pipeline(name="rating-prediction-pipeline")
def rating_prediction_pipeline(
    data_path: str,
    batch_input_path: str,
    batch_output_path: str,
    model_name: str,
    gcs_bucket: str = GCS_BUCKET
):
    """Clean rating prediction pipeline."""
    
    # Load training data
    load_task = load_data(data_path=data_path)
    
    # Train model
    train_task = train_model(input_data=load_task.outputs["output_data"])
    
    # Save model to GCS
    gcs_model_path = f"gs://{gcs_bucket}/models/{model_name}/model.pkl"
    save_task = save_model_to_gcs(
        model_input=train_task.outputs["model_output"],
        gcs_model_path=gcs_model_path
    )
    
    # Run batch prediction
    predict_task = batch_predict(
        gcs_model_path=save_task.output,
        input_path=batch_input_path,
        output_path=batch_output_path
    )
    
    # Validate results
    validate_task = validate_results(output_path=batch_output_path)
    
    # Set dependencies
    train_task.after(load_task)
    save_task.after(train_task)
    predict_task.after(save_task)
    validate_task.after(predict_task)

def compile_pipeline(output_file: str = "rating_prediction_pipeline.json"):
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=rating_prediction_pipeline,
        package_path=output_file
    )
    print(f"INFO: Pipeline compiled to {output_file}")

if __name__ == "__main__":
    compile_pipeline()