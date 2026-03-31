import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading data...")
    try:
        data = pd.read_csv('housing.csv')
    except Exception as e:
        print(f"Error loading housing.csv: {e}")
        return

    # Separate features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define numerical and categorical features
    numeric_features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ]
    categorical_features = ['ocean_proximity']

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the model pipeline using Linear Regression
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model
    print("Training the Linear Regression model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation Scores:")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save the model
    print("\nSaving the trained model pipeline to model.pkl...")
    joblib.dump(model_pipeline, 'model.pkl')
    print("Training complete!")

if __name__ == '__main__':
    main()
