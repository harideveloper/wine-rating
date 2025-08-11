# #!/usr/bin/env python3
# """Simple test for IS_LOCAL logic."""

# import os

# # Test the fixed approach
# IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"
# # IS_LOCAL = os.getenv("IS_LOCAL", "False")

# print(f"Environment variable IS_LOCAL: '{os.getenv('IS_LOCAL')}'")
# print(f"Converted to boolean: {IS_LOCAL}")

# if IS_LOCAL:
#     print("✅ LOCAL: Initializing AI Platform for local environment")
# else:
#     print("✅ CI/PROD: Initializing AI Platform for CI/production environment")


from google.cloud import aiplatform

# Set these to match your environment
project = "dev2-ea8f"
location = "europe-west2"  # or wherever your models are
model_display_name = "wine-quality-online-prediction-model"

def parse_metric(value_str: str) -> float:
    """Convert dashed format to float: '0-9992' -> 0.9992"""
    try:
        return (
            float(value_str.replace("-", "."))
            if "-" in value_str
            else float(value_str)
        )
    except (ValueError, TypeError):
        return 0.0

def fetch_latest_promotable_model():
    print(f"\nInitializing AI Platform for project={project}, location={location}")
    aiplatform.init(project=project, location=location)
    
    print(f"\nListing models with display_name='{model_display_name}'")
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')

    if not models:
        raise ValueError(f"No models found with display name: {model_display_name}")

    sorted_models = sorted(models, key=lambda m: m.create_time, reverse=True)

    print(f"\nFound {len(sorted_models)} models. Checking for 'ready-for-promotion=true'...\n")

    selected_model = None
    for i, model in enumerate(sorted_models):
        labels = model.labels or {}
        print(f"[{i+1}] Model: {model.resource_name}")
        print(f"     Created: {model.create_time}")
        print(f"     Labels: {labels}")

        if labels.get("ready-for-promotion", "").lower() == "true":
            selected_model = model
            print(f"✅ Selected model (ready for promotion): {model.resource_name}\n")
            break

    if not selected_model:
        raise ValueError(
            f"No models with 'ready-for-promotion=true' found among {len(sorted_models)} models"
        )

    labels = selected_model.labels or {}
    metadata = {
        "display_name": selected_model.display_name,
        "resource_name": selected_model.resource_name,
        "harness_build_id": labels.get("harness-build-id", "unknown"),
        "eval_status": labels.get("eval-status", "unknown"),
        "ready_for_promotion": labels.get("ready-for-promotion", "false"),
        "quality_score": str(parse_metric(labels.get("quality-score", "0"))),
        "mae": str(parse_metric(labels.get("mae", "0"))),
        "mse": str(parse_metric(labels.get("mse", "0"))),
        "r2_score": str(parse_metric(labels.get("r2-score", "0"))),
        "rmse": str(parse_metric(labels.get("rmse", "0"))),
    }

    print("📦 Extracted Metadata:")
    for k, v in metadata.items():
        print(f" - {k}: {v}")

    print("\n✅ Model fetch complete.")
    return selected_model.resource_name, metadata

if __name__ == "__main__":
    try:
        uri, metadata = fetch_latest_promotable_model()
        print(f"\nFetched model URI: {uri}")
    except Exception as e:
        print(f"\n❌ Error during model fetch: {e}")

