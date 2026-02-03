# src/train.py
"""
Simple ML training script for CI/CD demonstration
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    """Train a simple Random Forest classifier on Iris dataset"""
    print("Loading data...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return accuracy

if __name__ == '__main__':
    train_model()