import streamlit as st
from ml_model import DiseasePredictor
from chat_manager import ChatManager
from utils import init_upload_folder, save_uploaded_file, generate_pdf_diagnosis
import json
from datetime import datetime
import time
import os
# Initialize components
disease_predictor = DiseasePredictor()
chat_manager = ChatManager()
init_upload_folder()

def show_patient_dashboard():
    st.title("Patient Dashboard")
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.username}!")
        st.write("Quick Tips:")
        st.write("- Describe your symptoms in detail for better diagnosis")
        st.write("- Check your chat messages regularly")
        st.write("- Save or print your diagnosis reports")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Chat with Doctor", "My Profile"])
    
    with tab1:
        st.header("Symptom Checker")
        
        # Initialize session state variables if they don't exist
        if 'current_diagnosis' not in st.session_state:
            st.session_state.current_diagnosis = None
        if 'current_symptoms' not in st.session_state:
            st.session_state.current_symptoms = None
        if 'selected_disease' not in st.session_state:
            st.session_state.selected_disease = None
        if 'current_recommendations' not in st.session_state:
            st.session_state.current_recommendations = None
        if 'show_recommendations' not in st.session_state:
            st.session_state.show_recommendations = False
        if 'disease_options_map' not in st.session_state:
            st.session_state.disease_options_map = {}
        
        # Symptom input form
        symptoms = st.text_area(
            "Describe your symptoms in detail:",
            value=st.session_state.current_symptoms if st.session_state.current_symptoms else "",
            placeholder="Enter your symptoms here...",
            height=150
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Get Diagnosis"):
                if symptoms:
                    # Get disease prediction
                    diagnosis = disease_predictor.predict_disease(symptoms)
                    
                    if diagnosis:
                        # Store in session state
                        st.session_state.current_diagnosis = diagnosis
                        st.session_state.current_symptoms = symptoms
                        st.session_state.selected_disease = None  # Reset selection
                        st.session_state.current_recommendations = None  # Reset recommendations
                        st.session_state.show_recommendations = True  # Show the recommendations section
                        
                        # Create mapping between display options and disease names
                        st.session_state.disease_options_map = {
                            f"{disease} ({probability:.2%})": disease 
                            for disease, probability in diagnosis.items()
                        }
                    else:
                        st.warning("Could not generate diagnosis. Please try describing your symptoms differently.")
                else:
                    st.warning("Please enter your symptoms to get a diagnosis.")
        
        with col2:
            if st.button("Clear"):
                # Clear all session state
                st.session_state.current_diagnosis = None
                st.session_state.current_symptoms = None
                st.session_state.selected_disease = None
                st.session_state.current_recommendations = None
                st.session_state.show_recommendations = False
                st.session_state.disease_options_map = {}
                st.rerun()
        
        # Display results if we have a diagnosis
        if st.session_state.current_diagnosis and st.session_state.show_recommendations:
            st.subheader("Diagnosis Results")
            
            # Create radio buttons for diagnosis selection
            diagnosis_options = list(st.session_state.disease_options_map.keys())
            
            # Handle selection change
            def on_diagnosis_select():
                selected_option = st.session_state.diagnosis_selection
                # Get the actual disease name from the mapping
                disease_name = st.session_state.disease_options_map[selected_option]
                if disease_name != st.session_state.selected_disease:
                    st.session_state.selected_disease = disease_name
                    # Update recommendations for the newly selected disease
                    st.session_state.current_recommendations = disease_predictor.get_detailed_recommendations(
                        disease_name,
                        st.session_state.current_diagnosis[disease_name]
                    )
            
            # Radio buttons for diagnosis selection
            st.radio(
                "Select a diagnosis to view detailed recommendations:",
                diagnosis_options,
                key="diagnosis_selection",
                on_change=on_diagnosis_select
            )
            
            # Get selected disease name if not set
            if st.session_state.selected_disease is None and diagnosis_options:
                first_option = diagnosis_options[0]
                st.session_state.selected_disease = st.session_state.disease_options_map[first_option]
                st.session_state.current_recommendations = disease_predictor.get_detailed_recommendations(
                    st.session_state.selected_disease,
                    st.session_state.current_diagnosis[st.session_state.selected_disease]
                )
            
            # Display recommendations for selected disease
            if st.session_state.current_recommendations:
                with st.expander(f"Recommendations for {st.session_state.selected_disease}", expanded=True):
                    for rec in st.session_state.current_recommendations:
                        st.write(f"{rec}")
            
            # Create columns for action buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Save Selected Diagnosis"):
                    # Save the interaction to history with selected diagnosis
                    save_patient_interaction(
                        st.session_state.current_symptoms,
                        {st.session_state.selected_disease: st.session_state.current_diagnosis[st.session_state.selected_disease]},
                        st.session_state.current_recommendations
                    )
                    st.success(f"Diagnosis for {st.session_state.selected_disease} has been saved!")
            
            with col2:
                # Generate report with selected diagnosis
                filepath = generate_pdf_diagnosis(
                    st.session_state.username,
                    st.session_state.current_symptoms,
                    {st.session_state.selected_disease: st.session_state.current_diagnosis[st.session_state.selected_disease]},
                    st.session_state.current_recommendations
                )
                
                # Provide download button
                with open(filepath, "rb") as f:
                    st.download_button(
                        "Download Report",
                        f,
                        file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    with tab2:
        st.header("Chat with Doctor")
        
        # Get available doctors
        doctors = get_available_doctors()
        if not doctors:
            st.info("No doctors are currently available.")
            return
        
        # Create two columns for chat list and chat window
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Doctors")
            # Display doctors with their last message preview and unread count
            for doctor in doctors:
                last_message = chat_manager.get_last_message(st.session_state.username, doctor)
                unread_count = chat_manager.get_unread_count(st.session_state.username, doctor)
                preview = last_message['message'][:30] + "..." if last_message else "No messages yet"
                timestamp = last_message['timestamp'] if last_message else ""
                
                button_text = f"Dr. {doctor}"
                if unread_count > 0:
                    button_text += f" ({unread_count} unread)"
                button_text += f"\n{preview}\n{timestamp}"
                
                if st.button(button_text, key=f"doctor_{doctor}"):
                    st.session_state.selected_doctor = doctor
                    chat_manager.mark_as_read(st.session_state.username, doctor)
                    st.rerun()
        
        with col2:
            if 'selected_doctor' not in st.session_state:
                st.info("Select a doctor to start chatting")
            else:
                st.subheader(f"Chat with Dr. {st.session_state.selected_doctor}")
                
                # Chat messages container with custom styling
                chat_container = st.container()
                with chat_container:
                    st.markdown("""
                        <style>
                        .message {
                            padding: 10px;
                            margin: 5px;
                            border-radius: 10px;
                            max-width: 70%;
                        }
                        .sent {
                            background-color: #DCF8C6;
                            margin-left: auto;
                        }
                        .received {
                            background-color: #E8E8E8;
                            margin-right: auto;
                        }
                        .image-message {
                            max-width: 200px;
                            margin: 5px;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Get and display chat history
                    chat_history = chat_manager.get_chat_history(st.session_state.username, st.session_state.selected_doctor)
                    
                    for message in chat_history:
                        if message['type'] == 'image':
                            if message['sender'] == st.session_state.username:
                                st.markdown(f'<div class="message sent"><img src="{message["image_url"]}" class="image-message"><br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="message received"><img src="{message["image_url"]}" class="image-message"><br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                        else:
                            if message['sender'] == st.session_state.username:
                                st.markdown(f'<div class="message sent">{message["message"]}<br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="message received">{message["message"]}<br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                
                # Message input and image upload
                  # Message input and image upload
                with st.form(key='message_form'):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        new_message = st.text_input("Type your message:", key="new_message")
                    with col2:
                        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'gif'], key="image_upload")
                    
                    submit_button = st.form_submit_button("Send")
                    if submit_button and (new_message or uploaded_file):
                        image_path = None
                        if uploaded_file:
                            image_path = save_uploaded_file(uploaded_file)
                        
                        chat_manager.send_message(st.session_state.username, st.session_state.selected_doctor, new_message, image_path)
                        st.rerun()
    
    with tab3:
        st.header("My Profile")
        st.write("Profile management features will be implemented in the next phase.")
        st.write("You will be able to:")
        st.write("- Update your personal information")
        st.write("- View your medical history")
        st.write("- Manage your preferences")
        st.write("- Set notification preferences")

def get_available_doctors():
    """Get list of available doctors from the users file."""
    try:
        with open("users.json", 'r') as f:
            users = json.load(f)
        return [username for username, data in users.items() if data['role'] == 'doctor']
    except:
        return []

def save_patient_interaction(symptoms: str, diagnosis: dict, recommendations: list):
    """Save patient interaction to history."""
    try:
        with open("patient_history.json", 'r') as f:
            history = json.load(f)
    except:
        history = {}
    
    if st.session_state.username not in history:
        history[st.session_state.username] = []
    
    history[st.session_state.username].append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symptoms': symptoms,
        'diagnosis': diagnosis,
        'recommendations': recommendations
    })
    
    with open("patient_history.json", 'w') as f:
        json.dump(history, f, indent=4) 

if __name__ == "__main__":
    show_patient_dashboard()