import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_ingest():
    print("--- Testing Ingestion ---")
    payload = {
        "patient_id": "P-12345",
        "clinic_id": "clinic_A",
        "text_content": "Patient presents with severe migraine and sensitivity to light. History of hypertension."
    }
    response = requests.post(f"{BASE_URL}/ingest", data=payload)
    if response.status_code == 200:
        print("Success:", response.json())
        return True
    else: 
        print("Failed:", response.text)
        return False

def test_search():
    print("\n--- Testing Search ---")
    payload = {
        "query_text": "migraine treatment",
        "clinic_id": "clinic_A",
        "limit": 3
    }
    response = requests.post(f"{BASE_URL}/search", json=payload)
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results:")
        for res in results:
            print(f"- Score: {res['score']:.4f} | Text: {res['text_content']}")
    else:
        print("Failed:", response.text)

if __name__ == "__main__":
    if test_ingest():
        test_search()
