"""
Records Agent - Qdrant Only

All data storage is handled by Qdrant. No SQL database needed.
This file is kept for backwards compatibility but is a no-op.
"""

def get_db():
    """No-op for backwards compatibility"""
    yield None

def init_db():
    """No-op for backwards compatibility - Qdrant handles all storage"""
    pass
