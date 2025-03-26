from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from model import RNNModel

app = FastAPI()

# Load model
input_size = 2
hidden_size = 64   4
output_size = 2

model = RNNModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("rnn_weather_model.pth", map_location=torch.device("cpu")))
model.eval()

class SequenceInput(BaseModel):
    sequence: list[list[float]]

@app.post("/predict")
def predict(input_data: SequenceInput):
    try:
        data = torch.tensor([input_data.sequence], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(data).numpy()[0]
        return {
            "predicted_temperature": round(float(prediction[0]), 4),
            "predicted_humidity": round(float(prediction[1]), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
