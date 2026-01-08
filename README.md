# AI Disease Prediction System

An AI-driven disease prediction system with a chatbot interface, built using Streamlit and Python.

## Features

- User authentication with role-based access (patient/doctor)
- Patient dashboard for symptom input and disease prediction
- Doctor dashboard for viewing patient history
- AI-powered disease prediction based on symptoms using TF-IDF and cosine similarity
- Detailed medical recommendations based on predicted diagnoses
- Real-time chat functionality between doctors and patients
- Secure user data and chat history storage
- Patient history tracking with detailed symptom and diagnosis records

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage

### For Patients:
1. Register or log in as a patient
2. Use the Symptom Checker to:
   - Enter your symptoms in detail
   - Get AI-powered disease predictions
   - View detailed medical recommendations
3. Use the Chat with Doctor feature to:
   - Select an available doctor
   - Send and receive messages
   - View chat history

### For Doctors:
1. Register or log in as a doctor
2. Use the Patient History tab to:
   - View patient records
   - Access detailed symptom and diagnosis history
3. Use the Chat with Patients tab to:
   - Select a patient to chat with
   - Send and receive messages
   - View conversation history

## Technical Details

### Disease Prediction
- Uses TF-IDF vectorization for symptom analysis
- Implements cosine similarity for disease matching
- Provides probability scores for multiple potential diagnoses
- Offers detailed, disease-specific medical recommendations

### Chat System
- Real-time messaging between doctors and patients
- Persistent chat history storage
- User-friendly interface for both roles
- Secure message handling

### Data Storage
- JSON-based storage for:
  - User credentials
  - Patient history
  - Chat messages
- Secure password hashing
- Role-based access control

## Future Enhancements

- Integration with more advanced ML models
- Enhanced doctor-patient matching system
- Appointment scheduling
- Prescription management
- Medical image analysis
- Multi-language support
- Mobile application version

## Security Note

This is a prototype implementation. For production use, consider:
- Implementing a proper database system
- Adding more robust security measures
- Implementing proper data encryption
- Adding user session management
- Implementing proper API rate limiting
- Adding audit logging
- Implementing HIPAA compliance measures 