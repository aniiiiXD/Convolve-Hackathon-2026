from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
load_dotenv()
# 1. INITIALIZE CLIENT
# Connect to your local Qdrant instance
client = QdrantClient(url=os.getenv("ENDPOINT"),api_key=os.getenv("API_KEY"))

collection_name = "city_search"

# 2. CREATE COLLECTION
# A collection is a set of points (vectors + payload).
# We define the vector size (4 dimensions for this toy example) and distance metric.
if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4, 
        distance=models.Distance.COSINE
    ),
)

print(f"Collection '{collection_name}' created!")

# 3. ADD DATA (UPSERT)
# We upload "Points". Each point has an ID, a Vector, and a Payload (metadata)[cite: 27071].
# In a real app, vectors come from an embedding model (like OpenAI or HuggingFace).
# Here, we represent features manually: [cold, rain, architecture, food]
city_data = [
    models.PointStruct(
        id=1,
        vector=[0.1, 0.9, 0.8, 0.2], # High rain, high architecture
        payload={"city": "London", "country": "UK"}
    ),
    models.PointStruct(
        id=2,
        vector=[0.9, 0.1, 0.8, 0.5], # High cold, low rain
        payload={"city": "Berlin", "country": "Germany"}
    ),
    models.PointStruct(
        id=3,
        vector=[0.1, 0.1, 0.2, 0.9], # Low cold, high food
        payload={"city": "Rome", "country": "Italy"}
    )
]

client.upsert(
    collection_name=collection_name,
    points=city_data
)   

print("Data uploaded!")

# 4. SEARCH (QUERY)
# Let's find a city that is "Cold and has nice Architecture".
# We represent this query as a vector: [0.9, 0.1, 0.8, 0.1]
query_vector = [0.9, 0.1, 0.8, 0.1]

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=1  # Return the top 1 match 
) 

# 5. DISPLAY RESULTS
print("\n--- Search Result ---")
for result in search_result:
    print(f"Found City: {result.payload['city']}")
    print(f"Similarity Score: {result.score}")
    print(f"Metadata: {result.payload}")