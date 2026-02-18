import pickle

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

revenue_model = load_model("models/revenue_sarima.pkl")
sales_model = load_model("models/sales_sarima.pkl")
