# 🍷 Wine Rating Prediction System

This project implements an end-to-end machine learning pipeline for **wine rating prediction** using **Google Cloud Vertex AI**. It includes data preprocessing, model training, evaluation, deployment, and batch prediction capabilities.

---

## 📁 Project Structure

```
wine-recommendation-system/
│
├── pipeline/                      # ML pipeline components
│   ├── wine_rating_pipeline.py    # Main pipeline definition
│   └── run_wine_rating_pipeline.py # Script to trigger the pipeline
│
├── notebooks/                     # Jupyter notebooks for EDA
│   └── wine_rating_eda.ipynb      
│
├── dataset/                       
│   ├── raw/                       # Raw wine datasets
│   └── batch/                     # Batch prediction input/output
│
├── test/                          
│   ├── test_endpoint.py           # Python script to test endpoints
│   ├── test_endpoint.sh           # Shell script for endpoint testing
│   └── sample.json                # Sample prediction request
│
├── wine_venv/                     # Python virtual environment
├── .gitignore                     # Files to ignore in version control
└── Makefile                       # Automation scripts
```

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.9 or higher  
- Google Cloud SDK installed  
- GCP project with Vertex AI enabled  

### Environment Setup

```bash
# Create Python virtual environment
python -m venv wine_venv

# Activate the environment
source wine_venv/bin/activate  # On Windows: wine_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Using the Pipeline

### Running the Pipeline

```bash
cd pipeline
python run_wine_rating_pipeline.py
```

This will:
- Load and preprocess the wine data
- Train a RandomForest regression model
- Evaluate model performance

If the model meets the evaluation threshold, it will be:
- Registered in Vertex AI Model Registry  
- Deployed to an endpoint  
- Used for batch prediction  

---

## 🔍 Testing the Endpoint

```bash
cd test
./test_endpoint.sh
```

Or use Python:

```bash
python test/test_endpoint.py
```

---

## 📊 Wine Data Analysis

See `notebooks/wine_rating_eda.ipynb` for exploratory data analysis including:
- Wine rating distributions  
- Price vs Rating trends  
- Country and region insights  
- Grape variety comparisons  
- Style-based recommendations  

---

## 🧠 Model Details

- **Algorithm**: RandomForest Regressor  
- **Features**: Country, Region, Type, Style, Grape, price_numeric  
- **Output**: Predicted wine rating (1 to 5)  
- **Deployment**: Vertex AI endpoint  

---

## 💡 Style-Based Recommendations

This system also supports **style matching** to recommend wines similar to a user's preferred style.

---

## 👩‍💻 For Developers

### Adding New Features

1. Add your components in `wine_rating_pipeline.py`  
2. Modify the pipeline definition to include them  
3. Test everything before deployment  

### Data Format for Prediction

Expected JSON input:

```json
{
  "instances": [
    [price_numeric, "Country", "Region", "Type", "Style", "Grape"]
  ]
}
```

Example:

```json
{
  "instances": [
    [15.99, "France", "Burgundy", "White", "Rich & Toasty", "Chardonnay"]
  ]
}
```

---

## 🛠 Troubleshooting

- Make sure the prediction request format is correct  
- Verify model deployment status in Vertex AI Console  
- Double-check your GCP project and region configuration  

📑 Check the logs in the Vertex AI console for more information or contact project maintainers for help.

---
