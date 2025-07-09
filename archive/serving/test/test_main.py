# tests/test_main.py - Tests for the generic serving container
import pytest
import httpx
import json
import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

@pytest.fixture
def client():
    """Create test client."""
    with httpx.Client(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    # Create a simple pipeline
    X = np.array([[10.0, 0], [15.0, 1], [20.0, 0], [25.0, 1]])  # price, category
    y = np.array([3.5, 4.0, 4.2, 4.5])  # ratings
    
    # Simple pipeline
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', [0]),  # price
        ('cat', OneHotEncoder(handle_unknown='ignore'), [1])  # category
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

@pytest.fixture
def model_file(sample_model, tmp_path):
    """Save sample model to temporary file."""
    model_path = tmp_path / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(sample_model, f)
    return str(model_path)

def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Generic Model Serving"

def test_model_info_no_model(client):
    """Test model info endpoint when no model is loaded."""
    response = client.get("/model-info")
    assert response.status_code == 404

def test_predict_no_model(client):
    """Test prediction endpoint when no model is loaded."""
    payload = {
        "instances": [[10.0, 0], [15.0, 1]]
    }
    response = client.post("/predict", json=payload)
    # Should fail gracefully
    assert response.status_code in [503, 500]

@pytest.mark.asyncio
async def test_model_loading(model_file):
    """Test model loading functionality."""
    from main import load_model_from_path, model
    
    # Reset global model
    import main
    main.model = None
    
    # Test loading
    success = load_model_from_path(model_file)
    assert success is True
    assert main.model is not None

def test_predict_with_model(client, model_file):
    """Test prediction with a loaded model."""
    # Load model first
    from main import load_model_from_path
    load_model_from_path(model_file)
    
    # Test prediction
    payload = {
        "instances": [[10.0, 0], [15.0, 1]]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert all(isinstance(p, (int, float)) for p in data["predictions"])

def test_model_info_with_model(client, model_file):
    """Test model info endpoint with loaded model."""
    # Load model first
    from main import load_model_from_path
    load_model_from_path(model_file)
    
    response = client.get("/model-info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data

def test_invalid_prediction_format(client, model_file):
    """Test prediction with invalid input format."""
    # Load model first
    from main import load_model_from_path
    load_model_from_path(model_file)
    
    # Test with invalid format
    payload = {
        "instances": "invalid"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error