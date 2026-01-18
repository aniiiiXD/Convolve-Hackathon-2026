import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. LLM features will be disabled.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generates a response using Gemini 1.5 Pro.
        """
        if not self.model:
            return "Error: LLM is not configured (Missing GEMINI_API_KEY)."

        full_prompt = f"""
        You are MediSync, an advanced medical assistant.
        
        CONTEXT:
        {context}
        
        USER QUERY:
        {prompt}
        
        INSTRUCTIONS:
        Answer the query based on the context provided. Be professional, clinical, and concise.
        """
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini Generation Failed: {e}")
            return "I apologize, but I encountered an error generating a response."
