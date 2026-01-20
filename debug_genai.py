
try:
    from google import genai
    print("Import successful")
    client = genai.Client(api_key="TEST")
    print("Client init successful")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
