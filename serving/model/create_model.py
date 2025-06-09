#!/usr/bin/env python3
"""
create_sample_model.py - Create a simple wine rating prediction model for testing

This creates a trained model that predicts wine ratings based on:
- Price
- Country  
- Wine Type (Red/White)
- Grape Variety
- Style Description
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def create_sample_wine_data():
    """Create synthetic wine data for training."""
    np.random.seed(42)
    
    # Define categories
    countries = ['France', 'Italy', 'Spain', 'USA', 'Australia', 'Chile', 'Argentina']
    wine_types = ['Red', 'White', 'Rosé']
    grapes = ['Cabernet Sauvignon', 'Merlot', 'Pinot Noir', 'Chardonnay', 'Sauvignon Blanc', 
              'Riesling', 'Tempranillo', 'Sangiovese', 'Syrah', 'Grenache']
    styles = ['Rich', 'Light', 'Crisp', 'Smooth', 'Bold', 'Elegant', 'Fresh', 'Complex']
    
    n_samples = 1000
    data = []
    
    for i in range(n_samples):
        # Generate features
        price = np.random.lognormal(2.5, 0.8)  # Price distribution
        country = np.random.choice(countries)
        wine_type = np.random.choice(wine_types)
        grape = np.random.choice(grapes)
        style = np.random.choice(styles)
        
        # Generate rating based on features (with some logic)
        base_rating = 85  # Base wine rating
        
        # Price influence (diminishing returns)
        price_bonus = min(15, np.log(price) * 2)
        
        # Country influence
        country_bonus = {
            'France': 3, 'Italy': 2, 'Spain': 1, 'USA': 2,
            'Australia': 1, 'Chile': 0, 'Argentina': 0
        }.get(country, 0)
        
        # Wine type influence
        type_bonus = {'Red': 1, 'White': 0, 'Rosé': -1}.get(wine_type, 0)
        
        # Style influence
        style_bonus = {
            'Complex': 3, 'Elegant': 2, 'Rich': 1, 'Bold': 1,
            'Smooth': 0, 'Fresh': 0, 'Crisp': -1, 'Light': -1
        }.get(style, 0)
        
        # Calculate rating with noise
        rating = (base_rating + price_bonus + country_bonus + 
                 type_bonus + style_bonus + np.random.normal(0, 2))
        
        # Clamp rating between 80-100
        rating = max(80, min(100, rating))
        
        data.append({
            'price': round(price, 2),
            'country': country,
            'wine_type': wine_type,
            'grape': grape,
            'style': style,
            'rating': round(rating, 1)
        })
    
    return pd.DataFrame(data)

def train_wine_model(data, model_path='./model.pkl'):
    """Train wine rating prediction model."""
    print("🍷 Training wine rating prediction model...")
    
    # Prepare features and target
    features = ['price', 'country', 'wine_type', 'grape', 'style']
    X = data[features]
    y = data['rating']
    
    print(f"📊 Dataset: {len(data)} samples, {len(features)} features")
    print(f"📊 Target range: {y.min():.1f} - {y.max():.1f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    # Numeric features (just price)
    numeric_features = ['price']
    
    # Categorical features
    categorical_features = ['country', 'wine_type', 'grape', 'style']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create pipeline with Random Forest
    model = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train model
    print("🔄 Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"📊 Training RMSE: {train_rmse:.3f}")
    print(f"📊 Test RMSE: {test_rmse:.3f}")
    print(f"📊 Training R²: {train_r2:.3f}")
    print(f"📊 Test R²: {test_r2:.3f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {model_path}")
    
    # Test prediction with DataFrame (not list)
    sample_input_df = pd.DataFrame({
        'price': [25.0],
        'country': ['France'],
        'wine_type': ['Red'],
        'grape': ['Cabernet Sauvignon'],
        'style': ['Rich']
    })
    sample_pred = model.predict(sample_input_df)
    print(f"🧪 Sample prediction: {sample_pred[0]:.2f}")
    
    return model

def create_test_data():
    """Create test data for API testing."""
    test_cases = [
        # [price, country, wine_type, grape, style]
        [15.0, 'France', 'Red', 'Cabernet Sauvignon', 'Rich'],
        [30.0, 'Italy', 'Red', 'Sangiovese', 'Elegant'],
        [12.0, 'Spain', 'Red', 'Tempranillo', 'Bold'],
        [25.0, 'USA', 'White', 'Chardonnay', 'Complex'],
        [18.0, 'Australia', 'White', 'Sauvignon Blanc', 'Crisp'],
        [8.0, 'Chile', 'Red', 'Merlot', 'Smooth'],
        [45.0, 'France', 'Red', 'Pinot Noir', 'Elegant'],
        [20.0, 'Italy', 'White', 'Riesling', 'Fresh']
    ]
    
    return test_cases

def main():
    """Main function to create and save the model."""
    print("🍷 Creating sample wine rating prediction model...")
    
    # Create synthetic data
    print("📊 Generating synthetic wine data...")
    wine_data = create_sample_wine_data()
    
    # Save sample data
    wine_data.to_csv('./sample_wine_data.csv', index=False)
    print("💾 Sample data saved to sample_wine_data.csv")
    
    # Train model
    model = train_wine_model(wine_data, './model.pkl')
    
    # Create test data
    test_data = create_test_data()
    
    # Test the model with sample inputs
    print("\n🧪 Testing model with sample inputs:")
    for i, test_input in enumerate(test_data):
        # Convert to DataFrame for prediction
        test_df = pd.DataFrame({
            'price': [test_input[0]],
            'country': [test_input[1]],
            'wine_type': [test_input[2]],
            'grape': [test_input[3]],
            'style': [test_input[4]]
        })
        prediction = model.predict(test_df)
        price, country, wine_type, grape, style = test_input
        print(f"Test {i+1}: ${price} {country} {wine_type} {grape} ({style}) → {prediction[0]:.1f}")
    
    print("\n✅ Model creation completed!")
    print("\n📋 Files created:")
    print("   • model.pkl - Trained model")
    print("   • sample_wine_data.csv - Training data")
    
    print("\n🚀 Test your serving container:")
    print("1. Copy model.pkl to your serving directory")
    print("2. Run: make run")
    print("3. Test: curl -X POST http://localhost:8080/predict \\")
    print('     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"instances": {test_data[:2]}}}\'')

if __name__ == "__main__":
    main()