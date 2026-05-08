import os
import sys
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import math
import time

from fastapi import Query


# ========== إعداد المسارات ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ========== الاستيرادات ==========
try:
    from database import SessionLocal, engine, Base, Project, IoTData, RiskAlert, Equipment
    DB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Database not available: {e}")
    DB_AVAILABLE = False
    SessionLocal = None
    Base = None
    Project = None
    IoTData = None
    RiskAlert = None
    Equipment = None

try:
    from train_ai_model import ConstructionAIModel
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI Model not available: {e}")
    AI_AVAILABLE = False
    ConstructionAIModel = None

# ========== إنشاء الجداول ==========
if DB_AVAILABLE and Base is not None:
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
    except Exception as e:
        print(f"⚠️ Could not create tables: {e}")

# ========== التطبيق ==========
app = FastAPI(
    title="Construction AI API",
    description="AI for construction delay prediction & risk assessment",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== الـ AI ==========
ai_model = None

@app.on_event("startup")
async def startup():
    global ai_model
    
    if not AI_AVAILABLE:
        print("❌ AI module not available")
        return
    
    ai_model = ConstructionAIModel()
    models_dir = os.path.join(BASE_DIR, 'models')
    
    model_files = [
        'delay_classifier.pkl',
        'delay_regressor.pkl', 
        'risk_classifier.pkl'
    ]
    
    all_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)
    
    if all_exist:
        try:
            ai_model.load_models()
            print("✅ AI Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
    else:
        print(f"❌ Model files not found in {models_dir}")


# ========== Dependency ==========
def get_db():
    if not DB_AVAILABLE or SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========== النماذج (Pydantic) ==========

class EquipmentCreate(BaseModel):
    name: str
    type: str
    status: str
    productivity_factor: float

class ProjectCreate(BaseModel):
    owner_id: str
    name: str
    location: str
    city: str
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    bim_file_url: Optional[str] = None

class SensorInput(BaseModel):
    project_id: int
    temperature: float
    humidity: float
    water_level: Optional[float] = 0
    wind_speed: Optional[float] = 15
    helmet_lat: Optional[float] = None
    helmet_lng: Optional[float] = None
    machine_lat: Optional[float] = None
    machine_lng: Optional[float] = None

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


# ========== الروابط (API) ==========

@app.get("/")
def home():
    return {
        "message": "Construction AI API is running!",
        "ai_status": "loaded" if (ai_model and ai_model.is_trained) else "not_loaded",
        "version": "2.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_loaded": ai_model is not None and (ai_model.is_trained if ai_model else False),
        "db_available": DB_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


# ========== المعدات ==========

@app.get("/equipment")
def get_equipment(db: Session = Depends(get_db)):
    return db.query(Equipment).all()

@app.post("/equipment")
def create_equipment(equipment: EquipmentCreate, db: Session = Depends(get_db)):
    new = Equipment(**equipment.dict())
    db.add(new)
    db.commit()
    db.refresh(new)
    return new


# ========== المشاريع ==========

@app.post("/projects")
def add_project(project: ProjectCreate, db: Session = Depends(get_db)):
    new = Project(
        owner_id=project.owner_id,
        name=project.name,
        location=project.location,
        city=project.city,
        latitude=project.latitude,
        longitude=project.longitude,
        start_date=datetime.strptime(project.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(project.end_date, "%Y-%m-%d"),
        bim_file_url=project.bim_file_url,
        status="pending"
    )
    db.add(new)
    db.commit()
    db.refresh(new)
    return new

@app.get("/projects/owner/{owner_id}")
def get_owner_projects(owner_id: str, db: Session = Depends(get_db)):
    return db.query(Project).filter(Project.owner_id == owner_id).all()


# ========== IoT + AI ==========

@app.post("/sensors")
def receive_sensors(data: SensorInput, db: Session = Depends(get_db)):
    sensor = IoTData(
        project_id=data.project_id,
        temperature=data.temperature,
        humidity=data.humidity,
        water_level=data.water_level or 0,
        wind_speed=data.wind_speed or 15,
        helmet_lat=data.helmet_lat,
        helmet_lng=data.helmet_lng,
        machine_lat=data.machine_lat,
        machine_lng=data.machine_lng
    )
    db.add(sensor)
    db.commit()
    
    if ai_model and ai_model.is_trained:
        ai_input = {
            'temperature': data.temperature,
            'humidity': data.humidity,
            'rainfall': 1 if (data.water_level or 0) > 0.5 else 0,
            'pir': 1,
            'alert_count': 2 if data.temperature > 45 else 0,
            'equipment_availability': 0.8,
            'equipment_breakdown': 0,
            'planned_workers': 50,
            'actual_workers': 40,
            'activity_type': 'concrete',
            'planned_duration': 7,
            'complexity': 0.5,
            'project_type': 'residential'
        }
        
        result = ai_model.predict(ai_input)
        
        if result['risk']['level'] >= 2:
            alert = RiskAlert(
                project_id=data.project_id,
                risk_level=result['risk']['level'],
                risk_type='safety',
                description=f"🚨 {result['risk']['label']} - {result['risk']['action']}",
                confidence=result['risk']['confidence']
            )
            db.add(alert)
            db.commit()
        
        return {
            "sensor_saved": True,
            "ai_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    
    return {"sensor_saved": True, "ai_status": "not_available"}


# ========== AI Prediction (للـ Digital Twin) ==========

@app.post("/predict")
def predict_delay_risk(data: AIPredictionInput):
    if not ai_model or not ai_model.is_trained:
        raise HTTPException(status_code=503, detail="AI models not loaded. Check if .pkl files exist in /models/ folder")
    
    ai_input = {
        'temperature': data.temperature,
        'humidity': data.humidity,
        'rainfall': data.rainfall,
        'pir': data.pir,
        'alert_count': data.alert_count,
        'equipment_availability': data.equipment_availability,
        'equipment_breakdown': data.equipment_breakdown,
        'planned_workers': data.planned_workers,
        'actual_workers': data.actual_workers,
        'activity_type': data.activity_type,
        'planned_duration': data.planned_duration,
        'complexity': data.complexity,
        'project_type': data.project_type
    }
    
    result = ai_model.predict(ai_input)
    risk = result['risk']
    delay = result['delay']
    
    recommendations = []

    if data.temperature > 45:
        recommendations.append(
            "High temperature detected: delay concrete work or switch to night shifts"
        )

    if data.rainfall == 1:
        recommendations.append(
            "Rain detected: stop outdoor construction activities"
        )

    if data.equipment_breakdown == 1:
        recommendations.append(
            "Equipment breakdown detected: immediate maintenance required"
        )

    if data.equipment_availability < 0.7:
        recommendations.append(
            "Low equipment availability: review equipment allocation"
        )

    worker_shortage_pct = (
        (data.planned_workers - data.actual_workers)
        / data.planned_workers * 100
    ) if data.planned_workers > 0 else 0

    if worker_shortage_pct > 20:
        recommendations.append(
            f"Worker shortage detected ({worker_shortage_pct:.0f}%): hire additional workers"
        )

    if data.alert_count > 5:
        recommendations.append(
            "Multiple safety alerts detected: inspect the construction site"
        )

    if risk['risk_level'] == 'High':
        recommendations.append(
            "Critical risk level: emergency site inspection required"
        )

    elif risk['risk_level'] == 'Medium':
        recommendations.append(
            "Moderate risk detected: monitor site conditions carefully"
        )
    
    return {
        "success": True,
        "project_id": data.project_id,
        "delay": {
            "predicted": delay['predicted_delay'],
            "probability": delay['delay_probability'],
            "estimated_days": delay['predicted_delay_days'],
        },
        "risk": {
            "level": risk.get('risk_level', 'Unknown'),
            "label": risk['label'],
            "confidence": risk['confidence'],
            "action": risk['action'],
        },
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }


# ========== BIM Data ==========

@app.get("/projects/{project_id}/bim")
def get_bim_data(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    days_passed = (datetime.now() - project.start_date).days
    total_days = (project.end_date - project.start_date).days
    progress = min(100, (days_passed / total_days * 100)) if total_days > 0 else 0
    
    latest = db.query(IoTData).filter(
        IoTData.project_id == project_id
    ).order_by(IoTData.timestamp.desc()).first()
    
    return {
        "project_name": project.name,
        "bim_url": project.bim_file_url,
        "progress_percent": round(progress, 1),
        "status": project.status,
        "sensors": {
            "temperature": latest.temperature if latest else 30,
            "humidity": latest.humidity if latest else 50,
        }
    }

# ========== Live Equipment Data ==========

@app.get('/equipment/live')
def get_live_equipment(
    project_id: str = Query(...),
):

    move = math.sin(
        time.time() / 45,
    ) * 0.002

    return {
        'project_id': project_id,

        'equipment': [
            {
                'equipment_id':
                    'eq-crane-01',

                'name':
                    'Tower Crane',

                'type':
                    'Crane',

                'status':
                    'Working',

                'latitude':
                    21.543333 + move,

                'longitude':
                    39.172779 + move,

                'speed':
                    2.1,

                'image_url':
                    'https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=400',

                'last_update':
                    time.strftime(
                        '%Y-%m-%dT%H:%M:%S',
                    ),
            },

            {
                'equipment_id':
                    'eq-excavator-01',

                'name':
                    'Excavator',

                'type':
                    'Heavy Equipment',

                'status':
                    'Working',

                'latitude':
                    21.546333 - move,

                'longitude':
                    39.170779 + move,

                'speed':
                    1.4,

                'image_url':
                    'https://images.unsplash.com/photo-1581094288338-2314dddb7ece?w=400',

                'last_update':
                    time.strftime(
                        '%Y-%m-%dT%H:%M:%S',
                    ),
            },

            {
                'equipment_id':
                    'eq-bulldozer-01',

                'name':
                    'Bulldozer',

                'type':
                    'Heavy Equipment',

                'status':
                    'Maintenance',

                'latitude':
                    21.540333 + move,

                'longitude':
                    39.176779 - move,

                'speed':
                    0.0,

                'image_url':
                    'https://images.unsplash.com/photo-1517089596392-fb9a9033e05b?w=400',

                'last_update':
                    time.strftime(
                        '%Y-%m-%dT%H:%M:%S',
                    ),
            },
        ],
    }


# ========== تشغيل مباشر ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

