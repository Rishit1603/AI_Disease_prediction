import json
import os
import hashlib
from typing import Dict, Optional

# File to store user data
USERS_FILE = "users.json"

def load_users() -> Dict:
    """Load users from the JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users: Dict) -> None:
    """Save users to the JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password: str) -> str:
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str, role: str) -> bool:
    """
    Register a new user.
    Returns True if registration successful, False if username already exists.
    """
    users = load_users()
    
    if username in users:
        return False
    
    users[username] = {
        'password_hash': hash_password(password),
        'role': role
    }
    
    save_users(users)
    return True

def authenticate_user(username: str, password: str, role: str) -> bool:
    """
    Authenticate a user.
    Returns True if credentials are valid and role matches, False otherwise.
    """
    users = load_users()
    
    if username not in users:
        return False
    
    user = users[username]
    if user['password_hash'] != hash_password(password):
        return False
    
    if user['role'] != role:
        return False
    
    return True 