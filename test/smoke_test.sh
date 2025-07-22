
#!/bin/bash

# Configuration
ENDPOINT_ID="401185404696395776"
PROJECT_ID="212373574046"
DATA_FOLDER="winedata"

# Check if winedata directory exists
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: $DATA_FOLDER directory not found. Please create it and add JSON files."
    exit 1
fi

# Count JSON files
JSON_FILES=$(find "$DATA_FOLDER" -name "*.json" | sort)
FILE_COUNT=$(echo "$JSON_FILES" | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No JSON files found in $DATA_FOLDER directory."
    exit 1
fi

echo "Found $FILE_COUNT JSON files in $DATA_FOLDER directory."

# Process each JSON file
for JSON_FILE in $JSON_FILES; do
    FILENAME=$(basename "$JSON_FILE")
    WINE_NAME="${FILENAME%.json}"
    
    echo "------------------------------------------------------"
    
    # Display wine details from the file
    WINE_DATA=$(cat "$JSON_FILE" | grep -o '\[.*\]' | head -1)   
    RESPONSE=$(curl -s -X POST \
         -H "Authorization: Bearer $(gcloud auth print-access-token)" \
         -H "Content-Type: application/json" \
         "https://europe-west2-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west2/endpoints/${ENDPOINT_ID}:predict" \
         -d "@${JSON_FILE}")
    
    # Extract just the numeric prediction value
    PREDICTION=$(echo $RESPONSE | grep -o '"predictions":\s*\[\s*[0-9.]*\s*\]' | sed 's/"predictions":\s*\[\s*\([0-9.]*\)\s*\]/\1/')
    
    if [[ $RESPONSE == *"error"* ]]; then
        echo "❌ Error for $WINE_NAME wine"
        echo "Error message: $RESPONSE"
    else
        echo "✅ $WINE_NAME wine"
        echo "📊 Predicted rating: $PREDICTION / 5.0"
    fi
    
    echo "------------------------------------------------------"
    echo ""
    
    # Sleep briefly to avoid rate limiting
    sleep 1
done

echo "All wines processed."