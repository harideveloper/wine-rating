ENDPOINT_ID="8106439746848292864"
PROJECT_ID="212373574046"
INPUT_DATA_FILE="sample.json"

curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json" \
     "https://europe-west2-aiplatform.googleapis.com/v1/projects/${PROJECT_ID$/locations/europe-west2/endpoints/${ENDPOINT_ID}:predict" \
     -d "@${INPUT_DATA_FILE}"