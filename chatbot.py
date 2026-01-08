import json
import os
from datetime import datetime
from typing import Dict, List

# File to store patient history
PATIENT_HISTORY_FILE = "patient_history.json"

def predict_disease(symptoms: str) -> Dict[str, float]:
    """
    Predict potential diseases based on symptoms.
    This is a placeholder function that will be replaced with actual ML model integration.
    """
    # Placeholder implementation
    # In a real implementation, this would use the ML models from the GitHub repository
    return {
        "Common Cold": 0.75,
        "Flu": 0.60,
        "Allergies": 0.45
    }

def get_medical_recommendations(diagnosis: Dict[str, float]) -> List[str]:
    """
    Generate medical recommendations based on the diagnosis.
    This is a placeholder function that will be enhanced with more detailed recommendations.
    """
    recommendations = []
    
    for disease, probability in diagnosis.items():
        if probability > 0.5:
            if disease == "Common Cold":
                recommendations.extend([
                    "Get plenty of rest",
                    "Stay hydrated",
                    "Consider over-the-counter cold medicine",
                    "Use a humidifier"
                ])
            elif disease == "Flu":
                recommendations.extend([
                    "Seek medical attention if symptoms are severe",
                    "Get plenty of rest",
                    "Stay hydrated",
                    "Consider antiviral medication if prescribed"
                ])
            elif disease == "Allergies":
                recommendations.extend([
                    "Take antihistamines as directed",
                    "Avoid known allergens",
                    "Use nasal sprays if recommended",
                    "Consider allergy testing"
                ])
    
    return list(set(recommendations))  # Remove duplicates

def save_patient_interaction(username: str, symptoms: str, diagnosis: Dict[str, float]):
    """
    Save the patient's interaction for future reference.
    """
    history = {}
    if os.path.exists(PATIENT_HISTORY_FILE):
        with open(PATIENT_HISTORY_FILE, 'r') as f:
            history = json.load(f)
    
    if username not in history:
        history[username] = []
    
    history[username].append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symptoms': symptoms,
        'diagnosis': diagnosis
    })
    
    with open(PATIENT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4) 