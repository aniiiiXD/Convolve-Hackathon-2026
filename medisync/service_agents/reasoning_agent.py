from google import genai
from google.genai import types
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY not found. LLM features will be disabled.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                self.model_id = 'gemini-3-flash-preview'
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}")
                self.client = None

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generates a response using Gemini 3 Flash.
        """
        if not self.client:
            return "Error: LLM is not configured (Missing GEMINI_API_KEY)."

        system_instruction = """You are MediSync, an advanced medical assistant.
Be professional, clinical, and concise in your responses.
Answer queries based on the context provided."""

        user_content = f"""CONTEXT:
{context}

USER QUERY:
{prompt}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=user_content
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Generation Failed: {e}")
            return "I apologize, but I encountered an error generating a response."
