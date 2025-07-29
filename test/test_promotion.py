from google.cloud import aiplatform

def test_model_promotion():
    """Simple test to validate model promotion between registries."""
    
    # Configuration
    TEST_PROJECT = "dev2-ea8f"
    PROD_PROJECT = "dev1-bfa7" 
    LOCATION = "europe-west2"
    
    print("Testing model promotion between registries...")
    
    # Step 1: Get latest model from test registry
    print(f"\n1. Getting latest model from {TEST_PROJECT}...")
    aiplatform.init(project=TEST_PROJECT, location=LOCATION)
    
    models = aiplatform.Model.list()
    if not models:
        print("❌ No models found in test registry")
        return
    
    latest_model = max(models, key=lambda x: x.create_time)
    print(f"✅ Found model: {latest_model.display_name}")
    print(f"   Model ID: {latest_model.name}")
    
    # Step 2: Copy to production registry
    print(f"\n2. Promoting to {PROD_PROJECT}...")
    aiplatform.init(project=PROD_PROJECT, location=LOCATION)
    
    try:
        promoted_model = aiplatform.Model.upload(
            display_name=f"{latest_model.display_name}_prod",
            artifact_uri=latest_model.uri,
            serving_container_image_uri=latest_model.container_spec.image_uri,
            sync=True
        )
        
        print(f"✅ Model promoted successfully!")
        print(f"   Production model: {promoted_model.name}")
        
    except Exception as e:
        print(f"❌ Promotion failed: {str(e)}")

if __name__ == "__main__":
    test_model_promotion()