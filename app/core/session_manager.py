import uuid

class SessionManager: 
    """
    A simple in-memory manager that holds the converstation history.
    Each session is identified by a unique session ID and holds
    list of messages sent by user.
    """


    def __init__(self):
        self.sessions = {}

    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = [] # * starts with an empty convo list
        return session_id

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content:str):
        if session_id in self.sessions:
            self.sessions[session_id].append({"role": role, "content": content})
            return True
        
        return False
    

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
# Gloval instance for Session Manager
session_manager = SessionManager()