# House Price Prediction

An end-to-end Machine Learning project to predict house prices (`median_house_value`) using the California Housing dataset. The project demonstrates an automated machine learning pipeline performing data preprocessing and running a Linear Regression model.

## Setup

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script to load the `housing.csv` dataset, structure the preprocessing pipeline, train a `LinearRegression` model, and evaluate its performance.

```bash
python train.py
```
This script will produce a `model.pkl` file, which is the serialized representation of the entire preprocessing and prediction pipeline.

## Predicting House Prices

Use the prediction script by passing a JSON string containing the required features for a given location.

Example:
```bash
python predict.py '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0, "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0, "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}'
```
