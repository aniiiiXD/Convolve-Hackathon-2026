# MediSync Backend

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Ensure your `.env` file is in the project root (one level up) or in this directory.
   It must contain:
   ```
   QDRANT_URL=...
   QDRANT_API_KEY=...
   ```

## Running the Server

Run the following command from the `backend` directory:

```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, access the interactive API docs (Swagger UI) at:
[http://localhost:8000/docs](http://localhost:8000/docs)

## Verification
1. Open Swagger UI.
2. Use `/api/v1/ingest` to upload a test record.
3. Use `/api/v1/search` to query it.
