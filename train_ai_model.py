import joblib
import numpy as np
import os
from datetime import datetime

class ConstructionAIModel:
    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self.risk_model = None
        self.le_project_type = None
        self.le_activity_type = None
        self.le_risk_level = None
        self.le_alert_level = None
        self.le_season = None
        self.features = None
        self.is_trained = False
        
    def load_models(self):
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            print(f"🔍 Loading models from: {models_dir}")
            
            self.clf_model = joblib.load(os.path.join(models_dir, 'delay_classifier.pkl'))
            self.reg_model = joblib.load(os.path.join(models_dir, 'delay_regressor.pkl'))
            self.risk_model = joblib.load(os.path.join(models_dir, 'risk_classifier.pkl'))
            
            self.le_project_type = joblib.load(os.path.join(models_dir, 'le_project_type.pkl'))
            self.le_activity_type = joblib.load(os.path.join(models_dir, 'le_activity_type.pkl'))
            self.le_risk_level = joblib.load(os.path.join(models_dir, 'le_risk_level.pkl'))
            self.le_alert_level = joblib.load(os.path.join(models_dir, 'le_alert_level.pkl'))
            self.le_season = joblib.load(os.path.join(models_dir, 'le_season.pkl'))
            
            self.features = joblib.load(os.path.join(models_dir, 'feature_list.pkl'))
            
            self.is_trained = True
            print("✅ AI Models loaded!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.is_trained = False
    
    def prepare_features(self, data):
        planned_workers = data.get('planned_workers', 50)
        actual_workers = data.get('actual_workers', 40)
        worker_shortage = planned_workers - actual_workers
        worker_shortage_pct = (worker_shortage / planned_workers * 100) if planned_workers > 0 else 0
        worker_availability_pct = (actual_workers / planned_workers * 100) if planned_workers > 0 else 0
        
        temp = data.get('temperature', 35)
        humidity = data.get('humidity', 40)
        
        heat_index = temp + (humidity * 0.1)
        bad_weather = 1 if (temp > 45 or data.get('rainfall', 0) == 1 or humidity > 80) else 0
        critical_worker_shortage = 1 if worker_shortage_pct > 20 else 0
        equipment_sufficient = 1 if data.get('equipment_availability', 0.8) > 0.75 else 0
        
        alert_count = data.get('alert_count', 0)
        alert_level = 'Low' if alert_count <= 2 else 'Medium' if alert_count <= 5 else 'High'
        
        month = datetime.now().month
        season = 'Winter' if month in [12,1,2] else 'Spring' if month in [3,4,5] else 'Summer' if month in [6,7,8] else 'Autumn'
        
        try:
            activity_encoded = self.le_activity_type.transform([data.get('activity_type', 'concrete')])[0]
        except:
            activity_encoded = 0
        
        try:
            project_type_encoded = self.le_project_type.transform([data.get('project_type', 'residential')])[0]
        except:
            project_type_encoded = 0
        
        try:
            alert_level_encoded = self.le_alert_level.transform([alert_level])[0]
        except:
            alert_level_encoded = 0
            
        try:
            season_encoded = self.le_season.transform([season])[0]
        except:
            season_encoded = 0
        
        return np.array([[
            temp, humidity, data.get('rainfall', 0), data.get('pir', 1), alert_count,
            data.get('equipment_availability', 0.8), data.get('equipment_breakdown', 0),
            planned_workers, actual_workers, worker_shortage_pct,
            activity_encoded, data.get('planned_duration', 7), data.get('complexity', 0.5),
            heat_index, bad_weather, worker_availability_pct,
            critical_worker_shortage, equipment_sufficient,
            alert_level_encoded, season_encoded, project_type_encoded
        ]])
    
    def predict_risk(self, data):
        if not self.is_trained:
            return {'level': 0, 'label': 'Unknown', 'confidence': 0.0, 'action': 'AI not ready'}
        
        features_array = self.prepare_features(data)
        risk_pred = self.risk_model.predict(features_array)[0]
        risk_prob = self.risk_model.predict_proba(features_array)[0]
        risk_level = self.le_risk_level.inverse_transform([risk_pred])[0]
        
        risk_map = {
            'Low': {
                'level': 0,
                'label': 'Safe',
                'action': 'Continue construction work'
            },
            'Medium': {
                'level': 1,
                'label': 'Warning',
                'action': 'Review safety conditions'
            },
            'High': {
                'level': 2,
                'label': 'High Risk',
                'action': 'Stop work and inspect the site'
            }
        }
        result = risk_map.get(risk_level, {'level': 0, 'label': 'Unknown', 'action': 'No action'})
        result['confidence'] = round(float(risk_prob[risk_pred]), 3)
        result['risk_level'] = risk_level
        return result
    
    def predict_delay(self, data):
        if not self.is_trained:
            return {'predicted_delay': False, 'delay_probability': 0.0, 'predicted_delay_days': 0}
        
        features_array = self.prepare_features(data)
        delay_prob = self.clf_model.predict_proba(features_array)[0][1]
        delay_predicted = self.clf_model.predict(features_array)[0]
        
        if delay_predicted == 1:
            estimated_days = self.reg_model.predict(features_array)[0]
            estimated_days = max(1, min(10, estimated_days))
        else:
            estimated_days = 0
        
        return {
            'predicted_delay': bool(delay_predicted),
            'delay_probability': round(float(delay_prob), 3),
            'predicted_delay_days': round(float(estimated_days), 1)
        }
    
    def predict(self, data):
        return {
            'risk': self.predict_risk(data),
            'delay': self.predict_delay(data),
            'timestamp': datetime.now().isoformat()
        }