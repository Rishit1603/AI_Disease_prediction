import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

# Constants
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def init_upload_folder():
    """Initialize the upload folder if it doesn't exist."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save an uploaded file and return its path."""
    if uploaded_file is None:
        return None
    
    if not allowed_file(uploaded_file.name):
        st.error("Invalid file type. Please upload an image file.")
        return None
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath

def get_file_url(filepath: str) -> str:
    """Generate a data URL for the image file."""
    if not os.path.exists(filepath):
        return ""
    
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        extension = filepath.split('.')[-1].lower()
        return f"data:image/{extension};base64,{data}"

def generate_pdf_diagnosis(username: str, symptoms: str, diagnosis: dict, recommendations: list) -> str:
    """Generate a diagnosis report file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnosis_report_{timestamp}.txt"
    filepath = os.path.join("uploads", filename)
    
    # Replace emoji characters with text equivalents
    emoji_map = {
        '🏥': '[HOSPITAL]',
        '🔍': '[SYMPTOMS]',
        '⚠️': '[WARNING]',
        '⚡': '[ALERT]',
        '•': '-',
        '🔴': '[HIGH]',
        '🟡': '[MODERATE]',
        '🟢': '[LOW]',
        '⚕️': '[MEDICAL]'
    }
    
    def clean_text(text):
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        return text
    
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            # Write header
            f.write("=" * 50 + "\n")
            f.write("MEDICAL DIAGNOSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Patient information
            f.write("Patient Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Name: {username}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Symptoms
            f.write("Reported Symptoms:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{symptoms}\n\n")
            
            # Diagnosis
            f.write("Diagnosis:\n")
            f.write("-" * 20 + "\n")
            for disease, probability in diagnosis.items():
                f.write(f"- {disease} (Confidence: {probability:.2%})\n")
            f.write("\n")
            
            # Recommendations
            f.write("Medical Recommendations:\n")
            f.write("-" * 20 + "\n")
            for rec in recommendations:
                f.write(f"{clean_text(rec)}\n")
            
            # Disclaimer
            f.write("\n" + "=" * 50 + "\n")
            f.write("IMPORTANT DISCLAIMER:\n")
            f.write("This is an AI-generated diagnosis report and should not be considered as a substitute\n")
            f.write("for professional medical advice. Please consult with a healthcare provider for\n")
            f.write("proper medical diagnosis and treatment.\n")
            f.write("=" * 50 + "\n")
        
        return filepath
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        # Create a backup filepath in case of encoding issues
        backup_filepath = os.path.join("uploads", f"diagnosis_report_backup_{timestamp}.txt")
        with open(backup_filepath, "w") as f:
            f.write("Error generating detailed report. Please see the recommendations in the application.\n")
        return backup_filepath

def check_for_updates(last_check: datetime, chat_file: str) -> bool:
    """Check if there are any updates since the last check."""
    if not os.path.exists(chat_file):
        return False
    
    file_mtime = datetime.fromtimestamp(os.path.getmtime(chat_file))
    return file_mtime > last_check 