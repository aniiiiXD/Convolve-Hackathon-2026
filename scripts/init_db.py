import sys
import os

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medisync.core.db_sql import init_db
from medisync.services.auth import AuthService

def main():
    print("=== Initializing Database ===")
    init_db()
    print("✓ Tables created.")
    
    # Seed default users
    print("=== Seeding Users ===")
    AuthService.register_user("Dr_Strange", "DOCTOR", "Mayo_Clinic_01")
    AuthService.register_user("P-101", "PATIENT", "Mayo_Clinic_01")
    AuthService.register_user("P-102", "PATIENT", "Mayo_Clinic_01")
    print("✓ Users seeded.")

if __name__ == "__main__":
    main()
