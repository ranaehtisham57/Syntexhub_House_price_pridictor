import joblib
import pandas as pd
import sys
import json

def load_model(model_path='model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py '<json_data>'")
        print("Example (Make sure to wrap the JSON in single quotes):")
        example = {
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
            "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
            "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
        }
        print(f"python predict.py '{json.dumps(example)}'")
        sys.exit(1)

    json_data = sys.argv[1]
    
    try:
        data_dict = json.loads(json_data)
        # Convert to a pandas DataFrame containing a single row
        df = pd.DataFrame([data_dict])
    except Exception as e:
        print(f"Error parsing JSON input: {e}")
        sys.exit(1)
        
    model = load_model()
    
    try:
        prediction = model.predict(df)
        print(f"Predicted median house value: ${prediction[0]:,.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
