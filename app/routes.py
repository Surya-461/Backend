from fastapi import APIRouter
from app.model_loader import revenue_model
from app.model_loader import sales_model
# import pickle

router = APIRouter()

@router.get("/predict/revenue")
def predict_revenue(steps: int = 30):
    forecast = revenue_model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}

@router.get("/predict/sales")
def predict_sales(steps: int = 30):
    forecast = sales_model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}


# Load KMeans Model
# with open("models/kmeans_customer_segmentation_model.pkl", "rb") as f:
#     kmeans_model = pickle.load(f)


# @router.post("/segment")
# def segment_customer(income: float, spending_score: float):
    
#     input_data = np.array([[income, spending_score]])
    
#     cluster = kmeans_model.predict(input_data)[0]
    
#     return {
#         "income": income,
#         "spending_score": spending_score,
#         "segment": int(cluster)
#     }