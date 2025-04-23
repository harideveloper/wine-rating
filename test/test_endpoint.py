# test_endpoint.py
from google.cloud import aiplatform
import pandas as pd
import numpy as np


def test_endpoint():
    aiplatform.init(project="212373574046", location="europe-west2")
    endpoint = aiplatform.Endpoint("401185404696395776")
    test_data = pd.DataFrame({
        "price_numeric": [30.0],
        "Country": ["France"],
        "Region": ["Bordeaux"],
        "Type": ["Red"],
        "Style": ["Full-bodied"],
        "Grape": ["Cabernet Sauvignon"]
    })

    print("Sending data:", test_data)
    try:
        numpy_data = test_data.to_numpy()
        print("\nTrying numpy array format:")
        response = endpoint.predict(instances=numpy_data.tolist())
        print("Success! Response:", response)
        return
    except Exception as e:
        print("Failed:", str(e))

    try:
        numpy_data = test_data.to_numpy()
        print("\nTrying with feature names parameter:")
        response = endpoint.predict(
            instances=numpy_data.tolist(),
            parameters={"feature_names": list(test_data.columns)}
        )
        print("Success! Response:", response)
        return
    except Exception as e:
        print("Failed:", str(e))

    try:
        print("\nTrying dictionary format:")
        instances = test_data.to_dict(orient="records")
        response = endpoint.predict(instances=instances)
        print("Success! Response:", response)
        return
    except Exception as e:
        print("Failed:", str(e))

    print("\nAll attempts failed. You may need to redeploy with a custom prediction routine.")


if __name__ == "__main__":
    test_endpoint()
