import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import os

def main():
    parser = argparse.ArgumentParser(description="Predict Iris species from features.")
    parser.add_argument('--sepal-length', type=float, required=True, help="Sepal length in cm")
    parser.add_argument('--sepal-width', type=float, required=True, help="Sepal width in cm")
    parser.add_argument('--petal-length', type=float, required=True, help="Petal length in cm")
    parser.add_argument('--petal-width', type=float, required=True, help="Petal width in cm")
    
    args = parser.parse_args()
    
    model_path = 'flower_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run main.py first to train and save the model.")
        return
        
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    features = pd.DataFrame(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]],
        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    )
    
    prediction = model.predict(features)
    
    iris = load_iris()
    species = iris.target_names[prediction[0]]
    
    print(f"\nPredicted Species: {species.capitalize()} (class {prediction[0]})")
    
    # Not all models have predict_proba (like some SVC setups), but LR and DecisionTree do
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        print("\nClass Probabilities:")
        for idx, prob in enumerate(probabilities[0]):
            print(f"- {iris.target_names[idx].capitalize()}: {prob:.4f}")

if __name__ == '__main__':
    main()
