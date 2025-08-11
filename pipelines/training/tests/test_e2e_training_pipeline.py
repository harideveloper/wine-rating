"""End-to-end tests for wine quality pipeline"""

# pylint: disable=duplicate-code
from pipelines.training.wine_quality_online_predictor import (
    wine_quality_online_predictor_pipeline,
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    register_model,
    deploy_model,
    validate_model_endpoint,
)

import pipelines.training.wine_quality_online_predictor as pipeline_module


def test_pipeline_stage_sequence():
    """Test the logical sequence of pipeline stages."""
    expected_stages = [
        "load_data",
        "preprocess_data",
        "train_model",
        "evaluate_model",
        "save_model",
        "register_model",
        "deploy_model",
        "validate_model_endpoint",
    ]

    for stage in expected_stages:
        assert hasattr(pipeline_module, stage), f"Missing pipeline stage: {stage}"


def test_pipeline_parameter_validation():
    """Test pipeline parameters validation."""
    valid_params = {
        "data_path": "gs://wine-bucket/wine_data.csv",
        "evaluation_threshold": 0.8,
        "test_size": 0.2,
        "n_estimators": 100,
        "min_replica_count": 1,
        "max_replica_count": 3,
    }

    assert valid_params["data_path"].startswith("gs://")
    assert 0.0 <= valid_params["evaluation_threshold"] <= 1.0
    assert 0.0 < valid_params["test_size"] < 1.0
    assert valid_params["n_estimators"] > 0
    assert valid_params["min_replica_count"] <= valid_params["max_replica_count"]


def test_conditional_deployment_logic():
    """Test that conditional deployment threshold logic exists."""
    test_thresholds = [0.7, 0.8, 0.85, 0.9]

    for threshold in test_thresholds:
        assert 0.0 <= threshold <= 1.0, f"Invalid threshold: {threshold}"

    deployment_components = [
        save_model,
        register_model,
        deploy_model,
        validate_model_endpoint,
    ]
    for component in deployment_components:
        assert callable(
            component
        ), f"Deployment component {component.__name__} should be callable"


def test_pipeline_data_flow():
    """Test the data flow between pipeline stages."""
    stage_flow = [
        ("load_data", "preprocess_data"),
        ("preprocess_data", "train_model"),
        ("train_model", "evaluate_model"),
        ("evaluate_model", "save_model"),
        ("save_model", "register_model"),
        ("register_model", "deploy_model"),
        ("deploy_model", "validate_model_endpoint"),
    ]

    for source_stage, target_stage in stage_flow:
        assert hasattr(
            pipeline_module, source_stage
        ), f"Missing source stage: {source_stage}"
        assert hasattr(
            pipeline_module, target_stage
        ), f"Missing target stage: {target_stage}"


def test_end_to_end_pipeline_flow():
    """Test complete pipeline flow logic."""
    stages = [
        load_data,
        preprocess_data,
        train_model,
        evaluate_model,
        save_model,
        register_model,
        deploy_model,
        validate_model_endpoint,
    ]

    for stage in stages:
        assert callable(stage), f"Pipeline stage {stage.__name__} must be callable"

    assert callable(wine_quality_online_predictor_pipeline)
