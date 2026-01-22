"""
Authentication Service - Qdrant Only (No SQL)

Simple in-memory user store for hackathon demo.
In production, users would be stored in Qdrant with proper hashing.
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class UserRole(Enum):
    DOCTOR = "DOCTOR"
    PATIENT = "PATIENT"


@dataclass
class User:
    """User model"""
    id: str
    username: str
    role: UserRole
    clinic_id: str

    @property
    def user_id(self):
        return self.username


# In-memory user store for demo
# In production, this would be stored in Qdrant
DEMO_USERS = {
    "Dr_Strange": User(
        id="bdcf3928-848f-42a0-b0e1-c602d767dc15",
        username="Dr_Strange",
        role=UserRole.DOCTOR,
        clinic_id="Clinic-A"
    ),
    "Dr_House": User(
        id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        username="Dr_House",
        role=UserRole.DOCTOR,
        clinic_id="Clinic-B"
    ),
    "P-101": User(
        id="p101-uuid-1234-5678-90abcdef1234",
        username="P-101",
        role=UserRole.PATIENT,
        clinic_id="Clinic-A"
    ),
    "P-102": User(
        id="p102-uuid-1234-5678-90abcdef5678",
        username="P-102",
        role=UserRole.PATIENT,
        clinic_id="Clinic-A"
    ),
    "P-103": User(
        id="p103-uuid-1234-5678-90abcdef9012",
        username="P-103",
        role=UserRole.PATIENT,
        clinic_id="Clinic-B"
    ),
}


class AuthService:
    """Authentication service using in-memory store"""

    @staticmethod
    def login(username: str) -> Optional[User]:
        """
        Logs in a user by checking the in-memory store.
        Returns a User object or None.
        """
        user = DEMO_USERS.get(username)

        if user:
            print(f"[Auth] Logged in as {user.role.value}: {user.username} ({user.clinic_id})")
            return user
        else:
            print(f"[Auth] Login Failed: User '{username}' not found.")
            return None

    @staticmethod
    def register_user(username: str, role: str, clinic_id: str) -> User:
        """
        Register a new user (adds to in-memory store for this session).
        """
        if username in DEMO_USERS:
            return DEMO_USERS[username]

        import uuid
        new_user = User(
            id=str(uuid.uuid4()),
            username=username,
            role=UserRole(role),
            clinic_id=clinic_id
        )
        DEMO_USERS[username] = new_user
        print(f"[Auth] Registered new user: {username}")
        return new_user

    @staticmethod
    def get_user(username: str) -> Optional[User]:
        """Get user by username"""
        return DEMO_USERS.get(username)

    @staticmethod
    def list_users() -> list:
        """List all users"""
        return list(DEMO_USERS.values())
