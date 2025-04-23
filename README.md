# Wine Recommendation System

A machine learning system for wine recommendations built on Google Cloud Vertex AI.

## Overview

This project implements a content-based wine recommendation system using Vertex AI Pipelines. It takes a CSV of wine data, processes it, and builds a recommendation model that can suggest similar wines based on characteristics like region, type, vintage, and flavor profile.

### Features

- **End-to-End ML Pipeline**: Complete Vertex AI pipeline for data loading, preprocessing, training, evaluation, and deployment
- **Content-based Filtering**: Recommendations based on wine characteristics using cosine similarity
- **Web Interface**: User-friendly Streamlit app for interacting with recommendations
- **Automatic Deployment**: Seamless deployment to Vertex AI endpoints for production serving

## Architecture

The system consists of two main components:

1. **Vertex AI Pipeline**: Handles the entire ML workflow from data to deployment
2. **Streamlit Frontend**: Provides a web interface for users to interact with the recommendation system

### Pipeline Steps

1. **Data Loading**: Loads wine data from Google Cloud Storage
2. **Data Preprocessing**: Cleans data and extracts features from wine characteristics
3. **Model Training**: Trains a RandomForest regression model to predict wine ratings
4. **Model Evaluation**: Evaluates the model performance using metrics like RMSE and R²
5. **Model Registration**: Registers the model in Vertex AI Model Registry if it meets quality thresholds
6. **Model Deployment**: Deploys the model to a Vertex AI endpoint for real-time predictions


## Getting Started

### Prerequisites

- Google Cloud Platform account
- Vertex AI API enabled
- Google Cloud Storage bucket
- Python 3.9+

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/wine-recommendation-system.git
   cd wine-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   ```bash
   gcloud auth application-default login
   ```

### Data Preparation

Your wine dataset should be stored in Cloud Storage with the following recommended columns:

- **Title**: Wine name
- **Description**: Wine description text
- **Price**: Price of the wine
- **Capacity**: Bottle size
- **Grape**: Primary grape variety
- **Country**: Country of origin
- **Type**: Wine type (Red, White, Rosé)
- **ABV**: Alcohol content
- **Region**: Wine region
- **Style**: Wine style
- **Vintage**: Year of production

### Running the Pipeline

You can run the pipeline using the provided deployment script:

```bash
python pipeline/run_pipeline.py \
  --project-id=your-gcp-project-id \
  --bucket=your-gcs-bucket \
  --data-file=dataset/wine_v1.csv
```

Or using the Python API:

```python
from wine_recommendation_pipeline import run_pipeline

job = run_pipeline(
    project_id="your-project-id",
    gcs_bucket="your-gcs-bucket",
    data_path="gs://your-gcs-bucket/dataset/wine_v1.csv"
)
```

### Frontend App (To be Added)

```bash
cd app
python3 app.py
```

## Project Structure

```
wine-rating/
│
├── dataset/
│   ├── batch                           
│   └── raw               
│        └── wine_v1.csv
│
├── pipeline/
│   ├── wine_rating_pipeline.py 
│   └── run_pipeline.py               
│
├── app/
│   ├── app.py                           
│   └── requirements.txt                 
│
├── notebooks/
│   └── eda.ipynb         
│
├── scripts/
│   └── setup_environment.sh             
│
├── infra/
│   └── main.tf
│   └── iam.tf
│   └── vertexai.tf
│   └── provider.tf
│   └── variables.tf
│   └── terraform.tfvars
|
├── Makefile                            
├── requirements.txt                     
└── README.md                            
```

## Using the Makefile

The project includes a Makefile for common operations:

```bash
# Set up the environment
make setup

# Compile the pipeline
make pipeline

# Deploy the pipeline
PROJECT_ID=your-project-id BUCKET_NAME=your-bucket make deploy

# Run the frontend
make app
```

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and commit: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Vertex AI documentation and examples
- Streamlit for the frontend framework
- Scikit-learn for machine learning utilities