from medisync.core_agents.records_agent import get_db, init_db
from medisync.model_agents.data_models import User
from sqlalchemy.orm import Session
from typing import Optional

# Ensure tables exist
init_db()

class AuthService: 
    
    @staticmethod
    def login(username: str) -> Optional[User]:
        """
        Logs in a user by checking the Postgres database.
        Returns a User SQL object or None.
        """
        db: Session = next(get_db())
        user = db.query(User).filter(User.username == username).first()
        
        if user:
            print(f"[Auth] Logged in as {user.role}: {user.username} ({user.clinic_id})")
            return user
        else:
            print(f"[Auth] Login Failed: User '{username}' not found.")
            return None

    @staticmethod
    def register_user(username: str, role: str, clinic_id: str) -> User:
        """
        Helper to register a user for testing/setup.
        """
        db: Session = next(get_db())
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return existing
            
        new_user = User(username=username, role=role, clinic_id=clinic_id)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"[Auth] Registered new user: {username}")
        return new_user
