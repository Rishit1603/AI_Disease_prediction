import streamlit as st
import auth
from patient_dashboard import show_patient_dashboard
from doctor_dashboard import show_doctor_dashboard

def initialize_session_state():
    """Initialize all session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def main():
    st.set_page_config(
        page_title="AI Disease Prediction System",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Main application logic
    if not st.session_state.authenticated:
        show_login_page()
    else:
        if st.session_state.user_role == 'patient':
            show_patient_dashboard()
        elif st.session_state.user_role == 'doctor':
            show_doctor_dashboard()

def show_login_page():
    st.title("AI Disease Prediction System")
    
    # Create tabs for Login and Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["patient", "doctor"])
        
        if st.button("Login"):
            if auth.authenticate_user(username, password, role):
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        new_role = st.selectbox("Select Role", ["patient", "doctor"])
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                if auth.register_user(new_username, new_password, new_role):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

if __name__ == "__main__":
    main() 