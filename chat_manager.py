import json
import os
from datetime import datetime
from typing import Dict, List
from utils import save_uploaded_file, get_file_url

class ChatManager:
    def __init__(self):
        self.chat_file = "chats.json"
        self._initialize_chat_file()

    def _initialize_chat_file(self):
        """Initialize the chat file if it doesn't exist."""
        if not os.path.exists(self.chat_file):
            self._save_chats({})

    def _load_chats(self) -> Dict:
        """Load all chats from the JSON file."""
        if os.path.exists(self.chat_file):
            with open(self.chat_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_chats(self, chats: Dict) -> None:
        """Save all chats to the JSON file."""
        with open(self.chat_file, 'w') as f:
            json.dump(chats, f, indent=4)

    def get_chat_id(self, user1: str, user2: str) -> str:
        """Generate a unique chat ID for any two users."""
        # Sort usernames to ensure consistent chat ID regardless of order
        sorted_users = sorted([user1, user2])
        return f"{sorted_users[0]}_{sorted_users[1]}"

    def send_message(self, sender: str, receiver: str, message: str, image_path: str = None) -> None:
        """Send a message from sender to receiver."""
        chats = self._load_chats()
        chat_id = self.get_chat_id(sender, receiver)
        
        if chat_id not in chats:
            chats[chat_id] = []
        
        message_data = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'text'
        }
        
        if image_path:
            message_data['type'] = 'image'
            message_data['image_path'] = image_path
            message_data['image_url'] = get_file_url(image_path)
        
        chats[chat_id].append(message_data)
        self._save_chats(chats)

    def get_chat_history(self, user1: str, user2: str) -> List[Dict]:
        """Get chat history between two users."""
        chats = self._load_chats()
        chat_id = self.get_chat_id(user1, user2)
        messages = chats.get(chat_id, [])
        
        # Sort messages by timestamp
        return sorted(messages, key=lambda x: x['timestamp'])

    def get_user_chats(self, username: str) -> List[str]:
        """Get all chat partners for a user."""
        chats = self._load_chats()
        partners = set()
        
        for chat_id in chats:
            user1, user2 = chat_id.split('_')
            if user1 == username:
                partners.add(user2)
            elif user2 == username:
                partners.add(user1)
        
        return sorted(list(partners))

    def get_last_message(self, user1: str, user2: str) -> Dict:
        """Get the last message in a chat between two users."""
        chat_history = self.get_chat_history(user1, user2)
        return chat_history[-1] if chat_history else None

    def get_unread_count(self, username: str, partner: str) -> int:
        """Get the number of unread messages from a specific partner."""
        chat_history = self.get_chat_history(username, partner)
        return sum(1 for msg in chat_history if msg['sender'] == partner and not msg.get('read', False))

    def mark_as_read(self, username: str, partner: str) -> None:
        """Mark all messages from a partner as read."""
        chats = self._load_chats()
        chat_id = self.get_chat_id(username, partner)
        
        if chat_id in chats:
            for msg in chats[chat_id]:
                if msg['sender'] == partner:
                    msg['read'] = True
            
            self._save_chats(chats)