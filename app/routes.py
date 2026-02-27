from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
import os

from app.model_loader import (
    revenue_model,
    sales_model,
    customer_segmentation_model,
    return_prediction_model
)

router = APIRouter()


# =====================================
# 📊 REVENUE FORECAST
# =====================================
@router.get("/predict/revenue")
def predict_revenue(steps: int = 30):
    forecast = revenue_model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}


# =====================================
# 📦 SALES FORECAST
# =====================================
@router.get("/predict/sales")
def predict_sales(steps: int = 30):
    forecast = sales_model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}


# =====================================
# 👤 CUSTOMER SEGMENTATION
# =====================================

class CustomerData(BaseModel):
    total_spending: float = Field(..., ge=0)
    num_of_orders: float = Field(..., ge=0)
    average_order_value: float = Field(..., ge=0)
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)


@router.post("/predict/customer-segment")
def predict_customer_segment(data: CustomerData):

    input_df = pd.DataFrame([{
        "Total_Spending": data.total_spending,
        "num_of_orders": data.num_of_orders,
        "Average_Order_Value": data.average_order_value,
        "Recency": data.recency,
        "Frequency": data.frequency
    }])

    model = customer_segmentation_model
    predicted_cluster = model.predict(input_df)[0]

    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    recency_values = centers[:, 3]

    sorted_clusters = recency_values.argsort()

    cluster_label_map = {
        sorted_clusters[0]: "High Value Customer",
        sorted_clusters[1]: "Medium Value Customer",
        sorted_clusters[2]: "Low Value Customer"
    }

    return {
        "segment_number": int(predicted_cluster),
        "customer_type": cluster_label_map[predicted_cluster]
    }
    
    




class ReturnInput(BaseModel):
    Quantity: float
    Price: float
    Discount: float


@router.post("/predict-return")
def predict_return(data: ReturnInput):
    input_data = [[
        data.Quantity,
        data.Price,
        data.Discount
    ]]

    prediction = return_prediction_model.predict(input_data)

    return {"prediction": int(prediction[0])}


@router.get("/customers")
def get_customers():
    file_path = os.path.join("app", "data", "final_df.csv")
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


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