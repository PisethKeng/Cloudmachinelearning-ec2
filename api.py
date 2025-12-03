from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import boto3
from datetime import datetime
import uuid
import json

# Load model
with open("diabetes_best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class Patient(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

# --- S3 setup ---
S3_BUCKET = "your-bucket-name-here"

s3_client = boto3.client("s3", region_name="us-east-1")

def log_to_s3(input_data, prediction):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
    }

    key = f"predictions/{datetime.utcnow().date()}/{uuid.uuid4()}.json"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(record),
        ContentType="application/json",
    )

@app.post("/predict")
def predict(patient: Patient):

    df = pd.DataFrame([patient.dict()])

    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[:, 1][0])

    label = "Diabetic" if pred == 1 else "Non-diabetic"

    result = {"prediction": int(pred), "label": label, "diabetes_probability": prob}

    log_to_s3(patient.dict(), result)

    return result
