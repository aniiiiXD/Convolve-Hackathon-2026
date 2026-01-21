import os
# Force RERANKER active to ensure we use the path
os.environ["USE_RERANKER"] = "true"

import logging
logging.basicConfig(level=logging.INFO)

try:
    from medisync.model_agents.ranking_agent import get_reranker
    from medisync.core_agents.database_agent import client

    # print(f"Client API URL: {client.rest_uri}") # Invalid attribute
    
    reranker = get_reranker()
    print("Reranker initialized successfully using Shared Client.")
    
    # Simple check: Get collections
    colls = reranker.client.get_collections()
    print(f"Connected! Found {len(colls.collections)} collections.")
    for c in colls.collections:
        print(f" - {c.name}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
