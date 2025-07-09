# wine_quality_batch_predictor.py
"""
Wine Quality Batch Predictor Pipeline
Clean, simple ML pipeline with unique run tracking
"""

from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Dataset, Model, Input, Output, component
from constants import BASE_IMAGE, GCS_BUCKET, RANDOM_STATE, N_ESTIMATORS


@component(packages_to_install=["pandas", "google-cloud-storage"], base_image=BASE_IMAGE)
def load_data(data_path: str, output_data: Output[Dataset]):
    """Load CSV data from GCS."""
    import pandas as pd
    from google.cloud import storage
    import io
    import datetime
    import os
    import hashlib
    
    # Generate unique run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    context = f"{os.getpid()}_{os.environ.get('HOSTNAME', 'local')}"
    hash_suffix = hashlib.md5(context.encode()).hexdigest()[:6]
    run_id = f"wine_quality_{timestamp}_{hash_suffix}"
    
    # Load data
    bucket_name = data_path.replace('gs://', '').split('/')[0]
    blob_name = '/'.join(data_path.replace('gs://', '').split('/')[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Save data and run_id
    df.to_csv(output_data.path, index=False)
    output_data.metadata["run_id"] = run_id
    
    print(f"INFO: Run {run_id} - Loaded {len(df)} records")


@component(packages_to_install=["pandas==1.3.5", "scikit-learn==0.24.2", "numpy==1.21.6"], base_image=BASE_IMAGE)
def train_model(input_data: Input[Dataset], model_output: Output[Model], random_state: int, n_estimators: int):
    """Train wine quality prediction model."""
    import pandas as pd
    import numpy as np
    import pickle
    import re
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    run_id = input_data.metadata.get("run_id", "unknown")
    df = pd.read_csv(input_data.path)
    
    print(f"INFO: Run {run_id} - Training on {len(df)} records")
    print(f"INFO: Using random_state={random_state}, n_estimators={n_estimators}")
    
    # Prepare categorical data
    categorical_cols = ['Country', 'Type', 'Grape', 'Style']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Extract price
    def extract_price(price_str):
        if pd.isna(price_str):
            return 15.0
        matches = re.findall(r'(\d+\.?\d*)', str(price_str))
        return float(matches[0]) if matches else 15.0
    
    df["price_numeric"] = df["Price"].apply(extract_price) if "Price" in df.columns else 15.0
    
    # Create quality ratings with variation
    np.random.seed(random_state)  # Use parameter
    base_rating = 3.2 + (df['price_numeric'] * 0.03)
    
    country_bonus = df['Country'].map({
        'France': 0.5, 'Italy': 0.3, 'Spain': 0.2, 'Germany': 0.3, 'USA': 0.2, 'Australia': 0.1
    }).fillna(0.1)
    
    type_bonus = df['Type'].map({'Red': 0.2, 'White': 0.1, 'Sparkling': 0.3}).fillna(0.1)
    noise = np.random.normal(0, 0.2, len(df))
    
    df['Quality'] = np.clip(base_rating + country_bonus + type_bonus + noise, 3.0, 5.0)
    
    # Build and train model
    features = ['price_numeric'] + [col for col in categorical_cols if col in df.columns]
    
    transformers = [('num', 'passthrough', [0])]
    if len(features) > 1:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), list(range(1, len(features)))))
    
    pipeline = Pipeline([
        ('prep', ColumnTransformer(transformers, remainder='drop')),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))  # Use parameters
    ])
    
    pipeline.fit(df[features], df['Quality'])
    
    # Save model with metadata
    model_data = {'pipeline': pipeline, 'run_id': run_id, 'features': features}
    with open(model_output.path + ".pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    model_output.metadata["run_id"] = run_id
    print(f"INFO: Run {run_id} - Model trained successfully!")


@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def save_model(model_input: Input[Model], gcs_bucket: str, model_name: str) -> str:
    """Save model to GCS with unique path."""
    from google.cloud import storage
    
    run_id = model_input.metadata.get("run_id", "unknown")
    source_file = model_input.path + ".pkl"
    
    # Upload to unique path
    gcs_path = f"gs://{gcs_bucket}/models/{model_name}/{run_id}/wine_quality_model.pkl"
    
    client = storage.Client()
    bucket_name = gcs_bucket
    blob_name = f"models/{model_name}/{run_id}/wine_quality_model.pkl"
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file)
    
    print(f"INFO: Run {run_id} - Model saved to {gcs_path}")
    return gcs_path


@component(packages_to_install=[
    "pandas==1.3.5", "scikit-learn==0.24.2", "numpy==1.21.6", "google-cloud-storage"
], base_image=BASE_IMAGE)
def batch_predict(gcs_model_path: str, input_path: str, output_path: str) -> str:
    """Run batch predictions with unique output."""
    import pandas as pd
    import pickle
    import json
    import re
    import io
    import datetime
    import os
    import hashlib
    from google.cloud import storage
    
    # Generate NEW unique run ID for this batch prediction
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    context = f"{os.getpid()}_{os.environ.get('HOSTNAME', 'local')}"
    hash_suffix = hashlib.md5(context.encode()).hexdigest()[:6]
    batch_run_id = f"batch_{timestamp}_{hash_suffix}"
    
    client = storage.Client()
    
    # Load model
    model_parts = gcs_model_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(model_parts[0])
    blob = bucket.blob(model_parts[1])
    blob.download_to_filename("model.pkl")
    
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
        model = model_data['pipeline']
        training_run_id = model_data['run_id']  # Keep reference to training run
        features = model_data['features']
    
    print(f"INFO: Batch run {batch_run_id} using model from training run {training_run_id}")
    
    # Load input data
    input_parts = input_path.replace("gs://", "").split("/", 1)
    input_bucket = client.bucket(input_parts[0])
    input_blob = input_bucket.blob(input_parts[1])
    
    if input_path.endswith('.jsonl'):
        jsonl_data = input_blob.download_as_text()
        records = [json.loads(line) for line in jsonl_data.strip().split('\n') if line.strip()]
        df = pd.DataFrame(records)
    else:
        csv_data = input_blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
    
    # Prepare features
    categorical_cols = ['Country', 'Type', 'Grape', 'Style']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    def extract_price(price_str):
        if pd.isna(price_str):
            return 15.0
        matches = re.findall(r'(\d+\.?\d*)', str(price_str))
        return float(matches[0]) if matches else 15.0
    
    if "Price" in df.columns:
        df["price_numeric"] = df["Price"].apply(extract_price)
    elif "price_numeric" in df.columns:
        df["price_numeric"] = pd.to_numeric(df["price_numeric"], errors='coerce').fillna(15.0)
    else:
        df["price_numeric"] = 15.0
    
    # Predict
    X = df[features]
    predictions = model.predict(X)
    
    # Save results
    results = []
    for i, pred in enumerate(predictions):
        result = {
            "batch_run_id": batch_run_id,  
            "training_run_id": training_run_id,
            "wine_quality_score": round(float(pred), 2),
            "quality_grade": "Excellent" if pred >= 4.5 else "Good" if pred >= 4.0 else "Average",
            "model_path": gcs_model_path
        }
        for feature in features:
            result[f"input_{feature}"] = str(df.iloc[i][feature])
        results.append(result)
    
    # Upload results to outputs dir
    output_parts = output_path.replace("gs://", "").split("/", 1)
    output_bucket = client.bucket(output_parts[0])
    
    filename = f"wine_quality_predictions_{batch_run_id}.jsonl"
    with open(filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    blob_name = f"{output_parts[1]}/{filename}" if len(output_parts) > 1 else filename
    output_bucket.blob(blob_name).upload_from_filename(filename)
    
    final_path = f"gs://{output_parts[0]}/{blob_name}"
    print(f"INFO: Batch run {batch_run_id} - {len(predictions)} predictions saved to {final_path}")
    
    return final_path


@component(packages_to_install=["google-cloud-storage"], base_image=BASE_IMAGE)
def validate_results(output_path: str) -> str:
    """Validate prediction results."""
    from google.cloud import storage
    import json
    
    if not output_path.startswith('gs://'):
        return "Invalid output path"
    
    path_parts = output_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(path_parts[0])
    blob = bucket.blob(path_parts[1])
    
    data = blob.download_as_text()
    predictions = [json.loads(line) for line in data.strip().split('\n') if line.strip()]
    
    batch_run_id = predictions[0].get('batch_run_id', 'unknown') if predictions else 'unknown'
    training_run_id = predictions[0].get('training_run_id', 'unknown') if predictions else 'unknown'
    scores = [p['wine_quality_score'] for p in predictions]
    
    print(f"INFO: Batch run {batch_run_id} - Validated {len(predictions)} predictions")
    print(f"INFO: Used model from training run {training_run_id}")
    print(f"INFO: Score range: {min(scores):.2f} - {max(scores):.2f}")
    
    return f"Batch {batch_run_id}: {len(predictions)} predictions validated"


@dsl.pipeline(name="wine-quality-batch-predictor")
def wine_quality_batch_predictor(
    data_path: str,
    batch_input_path: str,
    batch_output_path: str,
    model_name: str,
    gcs_bucket: str = GCS_BUCKET
):
    """Wine Quality Batch Predictor Pipeline."""
    
    load_task = load_data(data_path=data_path)
    
    train_task = train_model(
        input_data=load_task.outputs["output_data"],
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS
    )
    
    save_task = save_model(
        model_input=train_task.outputs["model_output"],
        gcs_bucket=gcs_bucket,
        model_name=model_name
    )
    
    predict_task = batch_predict(
        gcs_model_path=save_task.output,
        input_path=batch_input_path,
        output_path=batch_output_path
    )
    
    validate_task = validate_results(output_path=predict_task.output)
    
    # Set dependencies
    train_task.after(load_task)
    save_task.after(train_task)
    predict_task.after(save_task)
    validate_task.after(predict_task)


def compile_pipeline(output_file: str = "wine_quality_batch_predictor.json"):
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=wine_quality_batch_predictor,
        package_path=output_file
    )
    print(f"INFO: Wine Quality Batch Predictor compiled to {output_file}")


if __name__ == "__main__":
    compile_pipeline()