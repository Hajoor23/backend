import os
import sys
import math
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from train_ai_model import ConstructionAIModel
    AI_AVAILABLE = True
except ImportError as e:
    print(f"AI Model not available: {e}")
    AI_AVAILABLE = False
    ConstructionAIModel = None

app = FastAPI(
    title="Construction AI API",
    description="AI for construction delay prediction, risk assessment, and equipment tracking simulation",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_model = None


@app.on_event("startup")
async def startup():
    global ai_model

    if not AI_AVAILABLE:
        print("AI module not available")
        return

    ai_model = ConstructionAIModel()
    models_dir = os.path.join(BASE_DIR, "models")

    model_files = [
        "delay_classifier.pkl",
        "delay_regressor.pkl",
        "risk_classifier.pkl",
    ]

    all_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)

    if all_exist:
        try:
            ai_model.load_models()
            print("AI Models loaded successfully!")
        except Exception as e:
            print(f"Error loading AI models: {e}")
    else:
        print(f"Model files not found in {models_dir}")


class AIPredictionInput(BaseModel):
    project_id: int
    temperature: float
    humidity: float
    rainfall: int = 0
    pir: int = 1
    alert_count: int = 0
    equipment_availability: float = 0.8
    equipment_breakdown: int = 0
    planned_workers: int = 50
    actual_workers: int = 40
    activity_type: str = "concrete"
    planned_duration: int = 7
    complexity: float = 0.5
    project_type: str = "residential"


@app.get("/")
def home():
    return {
        "message": "Construction AI API is running!",
        "ai_status": "loaded" if (ai_model and ai_model.is_trained) else "not_loaded",
        "version": "2.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_loaded": ai_model is not None and (ai_model.is_trained if ai_model else False),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
def predict_delay_risk(data: AIPredictionInput):
    if not ai_model or not ai_model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="AI models not loaded. Check if .pkl files exist in /models folder.",
        )

    ai_input = {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "rainfall": data.rainfall,
        "pir": data.pir,
        "alert_count": data.alert_count,
        "equipment_availability": data.equipment_availability,
        "equipment_breakdown": data.equipment_breakdown,
        "planned_workers": data.planned_workers,
        "actual_workers": data.actual_workers,
        "activity_type": data.activity_type,
        "planned_duration": data.planned_duration,
        "complexity": data.complexity,
        "project_type": data.project_type,
    }

    result = ai_model.predict(ai_input)
    risk = result["risk"]
    delay = result["delay"]

    recommendations = []

    if data.temperature > 45:
        recommendations.append("High temperature detected: delay concrete work or switch to night shifts.")

    if data.rainfall == 1:
        recommendations.append("Rain detected: stop outdoor construction activities.")

    if data.equipment_breakdown == 1:
        recommendations.append("Equipment breakdown detected: immediate maintenance required.")

    if data.equipment_availability < 0.7:
        recommendations.append("Low equipment availability: review equipment allocation.")

    worker_shortage_pct = (
        (data.planned_workers - data.actual_workers)
        / data.planned_workers
        * 100
    ) if data.planned_workers > 0 else 0

    if worker_shortage_pct > 20:
        recommendations.append(
            f"Worker shortage detected ({worker_shortage_pct:.0f}%): hire additional workers."
        )

    if data.alert_count > 5:
        recommendations.append("Multiple safety alerts detected: inspect the construction site.")

    if risk.get("risk_level") == "High":
        recommendations.append("Critical risk level: emergency site inspection required.")
    elif risk.get("risk_level") == "Medium":
        recommendations.append("Moderate risk detected: monitor site conditions carefully.")

    return {
        "success": True,
        "project_id": data.project_id,
        "delay": {
            "predicted": delay["predicted_delay"],
            "probability": delay["delay_probability"],
            "estimated_days": delay["predicted_delay_days"],
        },
        "risk": {
            "level": risk.get("risk_level", "Unknown"),
            "label": risk["label"],
            "confidence": risk["confidence"],
            "action": risk["action"],
        },
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/equipment/live")
def get_live_equipment(
    project_id: str = Query(...),
    lat: float = Query(21.5433),
    lng: float = Query(39.1728),
):
    now = time.time()

    crane_move = math.sin(now / 35) * 0.00055
    excavator_move = math.cos(now / 28) * 0.00065
    bulldozer_move = math.sin(now / 42) * 0.00045
    pump_move = math.cos(now / 38) * 0.00050

    return {
        "project_id": project_id,
        "site_center": {
            "latitude": lat,
            "longitude": lng,
        },
        "equipment": [
            {
                "equipment_id": "eq-crane-01",
                "name": "Tower Crane",
                "type": "Crane",
                "status": "Working",
                "latitude": lat + crane_move,
                "longitude": lng + (crane_move / 2),
                "speed": 1.2,
                "fuel_level": 78,
                "engine_status": "Running",
                "engine_temperature": 84,
                "engine_hours": 1240,
                "fault_code": None,
                "maintenance_alert": False,
                "image_url": "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=400",
                "last_update": datetime.now().isoformat(),
            },
            {
                "equipment_id": "eq-excavator-01",
                "name": "Excavator",
                "type": "Heavy Equipment",
                "status": "Working",
                "latitude": lat - 0.00035 + excavator_move,
                "longitude": lng + 0.00045 - (excavator_move / 2),
                "speed": 2.4,
                "fuel_level": 64,
                "engine_status": "Running",
                "engine_temperature": 91,
                "engine_hours": 870,
                "fault_code": None,
                "maintenance_alert": False,
                "image_url": "https://images.unsplash.com/photo-1581094288338-2314dddb7ece?w=400",
                "last_update": datetime.now().isoformat(),
            },
            {
                "equipment_id": "eq-bulldozer-01",
                "name": "Bulldozer",
                "type": "Heavy Equipment",
                "status": "Maintenance",
                "latitude": lat + 0.00050 - bulldozer_move,
                "longitude": lng - 0.00035 + bulldozer_move,
                "speed": 0.0,
                "fuel_level": 38,
                "engine_status": "Idle",
                "engine_temperature": 76,
                "engine_hours": 1510,
                "fault_code": "MNT-204",
                "maintenance_alert": True,
                "image_url": "https://images.unsplash.com/photo-1517089596392-fb9a9033e05b?w=400",
                "last_update": datetime.now().isoformat(),
            },
            {
                "equipment_id": "eq-pump-01",
                "name": "Water Pump",
                "type": "Utility",
                "status": "Paused",
                "latitude": lat - 0.00045 + (pump_move / 2),
                "longitude": lng - 0.00055 + pump_move,
                "speed": 0.0,
                "fuel_level": 52,
                "engine_status": "Stopped",
                "engine_temperature": 45,
                "engine_hours": 340,
                "fault_code": None,
                "maintenance_alert": False,
                "image_url": "https://images.unsplash.com/photo-1621905252507-b35492cc74b4?w=400",
                "last_update": datetime.now().isoformat(),
            },
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
