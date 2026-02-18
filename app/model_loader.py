import os
import pickle
import joblib

# Get absolute path of models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_pickle_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def load_joblib_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


# Load SARIMA models
revenue_model = load_pickle_model("revenue_sarima.pkl")
sales_model = load_pickle_model("sales_sarima.pkl")

# Load Customer Segmentation Pipeline
customer_segmentation_model = load_joblib_model(
    "customer_segmentation_pipeline.pkl"
)

# Load Return Prediction Model (XGBoost)
return_prediction_model = load_joblib_model("xgboost_return_model.pkl")
