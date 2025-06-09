# main.py - 100% generic model serving container (no domain-specific logic)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Generic Model Serving", version="1.0.0")

# Global model variable
model = None

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    instances: List[List[Any]]

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    predictions: List[float]

def load_local_model():
    """Load model from local file."""
    global model
    
    # Try common local paths
    model_paths = [
        "./model.pkl",
        "./models/model.pkl",
        "/app/model.pkl"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"📥 Loading model from {path}")
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("✅ Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                continue
    
    logger.warning("⚠️ No model found in local paths")
    return False

def smart_predict(model, instances):
    """Generic prediction that tries different input formats."""
    
    logger.info(f"🔍 Input: {len(instances)} instances, {len(instances[0])} features each")
    
    # Method 1: Try DataFrame (for models with categorical/mixed features)
    try:
        first_instance = instances[0]
        has_strings = any(isinstance(val, str) for val in first_instance)
        
        if has_strings:
            # Create DataFrame with generic column names
            num_features = len(first_instance)
            columns = [f'feature_{i}' for i in range(num_features)]
            df = pd.DataFrame(instances, columns=columns)
            logger.info(f"📊 Trying DataFrame with columns: {columns}")
            predictions = model.predict(df)
            logger.info("✅ DataFrame format worked")
            return predictions
            
    except Exception as df_error:
        logger.info(f"📊 DataFrame failed: {str(df_error)[:100]}...")
    
    # Method 2: Try numpy array (for numerical models)
    try:
        instances_array = np.array(instances)
        logger.info(f"📊 Trying numpy array shape: {instances_array.shape}")
        predictions = model.predict(instances_array)
        logger.info("✅ Numpy array format worked")
        return predictions
        
    except Exception as array_error:
        logger.info(f"📊 Numpy array failed: {str(array_error)[:100]}...")
    
    # Method 3: Try raw list (fallback)
    try:
        logger.info("📊 Trying raw list format")
        predictions = model.predict(instances)
        logger.info("✅ Raw list format worked")
        return predictions
        
    except Exception as list_error:
        logger.info(f"📊 Raw list failed: {str(list_error)[:100]}...")
    
    # Debug info if all methods fail
    logger.error(f"🚨 All formats failed. Model type: {type(model)}")
    if hasattr(model, 'feature_names_in_'):
        logger.error(f"🚨 Model trained with: {list(model.feature_names_in_)}")
    if hasattr(model, 'n_features_in_'):
        logger.error(f"🚨 Model expects {model.n_features_in_} features")
    
    raise Exception("Model prediction failed with all input formats")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("🚀 Starting generic model serving...")
    load_local_model()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generic prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        instances = request.instances
        logger.info(f"📊 Received prediction request: {len(instances)} instances")
        
        # Use smart prediction to handle different formats
        predictions = smart_predict(model, instances)
        
        # Convert to list of floats
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)
        
        predictions_list = [float(p) for p in predictions_list]
        
        logger.info(f"✅ Returned {len(predictions_list)} predictions")
        return PredictionResponse(predictions=predictions_list)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Generic Model Serving",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": ["/health", "/predict"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)