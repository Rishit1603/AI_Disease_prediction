import streamlit as st
from chat_manager import ChatManager
import json
import os
from typing import Dict, List
from datetime import datetime

# File to store patient history
PATIENT_HISTORY_FILE = "patient_history.json"

# Initialize chat manager
chat_manager = ChatManager()

def load_patient_history() -> Dict:
    """Load patient history from JSON file."""
    if os.path.exists(PATIENT_HISTORY_FILE):
        with open(PATIENT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_patient_history(history: Dict) -> None:
    """Save patient history to JSON file."""
    with open(PATIENT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def show_doctor_dashboard():
    st.title("Doctor Dashboard")
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.write(f"Welcome, Dr. {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["Patient History", "Chat with Patients"])
    
    with tab1:
        st.header("Patient History")
        
        # Load patient history
        try:
            with open("patient_history.json", 'r') as f:
                history = json.load(f)
        except:
            history = {}
        
        if not history:
            st.info("No patient history available yet.")
        else:
            # Get list of patients
            patients = list(history.keys())
            selected_patient = st.selectbox("Select a patient:", patients)
            
            if selected_patient:
                st.subheader(f"History for {selected_patient}")
                for record in history[selected_patient]:
                    with st.expander(f"Visit on {record['date']}"):
                        st.write("**Symptoms:**")
                        st.write(record['symptoms'])
                        
                        st.write("**Diagnosis:**")
                        for disease, probability in sorted(record['diagnosis'].items(), key=lambda x: x[1], reverse=True):
                            st.write(f"- {disease}: {probability:.2%}")
    
    with tab2:
        st.header("Chat with Patients")
        
        # Get patients with chat history
        patients = chat_manager.get_user_chats(st.session_state.username)
        if not patients:
            st.info("No active chats with patients.")
            return
        
        # Create two columns for chat list and chat window
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Patients")
            # Display patients with their last message preview
            for patient in patients:
                last_message = chat_manager.get_last_message(st.session_state.username, patient)
                preview = last_message['message'][:30] + "..." if last_message else "No messages yet"
                timestamp = last_message['timestamp'] if last_message else ""
                
                if st.button(f"{patient}\n{preview}\n{timestamp}", key=f"patient_{patient}"):
                    st.session_state.selected_patient = patient
                    st.rerun()
        
        with col2:
            if 'selected_patient' not in st.session_state:
                st.info("Select a patient to start chatting")
            else:
                st.subheader(f"Chat with {st.session_state.selected_patient}")
                
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
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Get and display chat history
                    chat_history = chat_manager.get_chat_history(st.session_state.username, st.session_state.selected_patient)
                    
                    for message in chat_history:
                        if message['sender'] == st.session_state.username:
                            st.markdown(f'<div class="message sent">{message["message"]}<br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="message received">{message["message"]}<br><small>{message["timestamp"]}</small></div>', unsafe_allow_html=True)
                
                # Method 1: Using a form
                with st.form(key='message_form'):
                    new_message = st.text_input("Type your message:", key="new_message")
                    submit_button = st.form_submit_button("Send")
                    if submit_button and new_message:
                        chat_manager.send_message(st.session_state.username, st.session_state.selected_patient, new_message)
                        st.rerun()

                # OR Method 2: Using callback
                # Define this at the top of your file or before the widget
                # def clear_message():
                #     if 'new_message' in st.session_state:
                #         st.session_state.new_message = ""

                # # Then in your UI
                # new_message = st.text_input("Type your message:", key="new_message", on_change=clear_message)
                # if st.button("Send"):
                #     if new_message:
                #         chat_manager.send_message(st.session_state.username, st.session_state.selected_patient, new_message)
                #         st.rerun() 