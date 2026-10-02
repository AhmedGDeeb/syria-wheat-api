import numpy as np
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Syria Wheat Yield Predictor")
model = joblib.load("wheat_model.pkl")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    rainfall: float = Form(...),
    solar: float = Form(...),
):
    pred = model.predict(np.array([[rainfall, solar]]))[0]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": f"{pred:,.0f} kg/ha",
            "rainfall": rainfall,
            "solar": solar,
        },
    )


@app.post("/api/predict")
async def predict_api(rainfall: float, solar: float):
    """Pure JSON API endpoint."""
    pred = model.predict(np.array([[rainfall, solar]]))[0]
    return {
        "rainfall_mm": rainfall,
        "solar_kwh_m2": solar,
        "predicted_yield_kg_ha": round(float(pred), 1),
    }