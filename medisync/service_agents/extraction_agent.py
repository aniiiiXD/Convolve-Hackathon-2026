"""
Medical Entity Extraction using Gemini

Extracts structured medical entities from clinical notes:
- Conditions/diagnoses
- Treatments/procedures
- Outcomes
- Duration
- Body parts/locations
"""

import json
import logging
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types
import os

logger = logging.getLogger(__name__)


class MedicalEntityExtractor:
    """Extract structured medical entities from clinical text"""

    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"

    def extract_entities(
        self,
        clinical_text: str,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Extract medical entities from clinical text

        Args:
            clinical_text: Clinical note text
            max_retries: Maximum retry attempts

        Returns:
            Dictionary of extracted entities or None
        """
        system_instruction = """You are a medical entity extraction system.
Extract structured medical information and return ONLY valid JSON."""

        prompt = f"""Extract structured medical information from the following clinical note.
Return a JSON object with these fields:

- condition: Primary medical condition/diagnosis (string)
- treatment: Primary treatment or procedure (string)
- outcome: Treatment outcome (e.g., "healed", "improving", "stable", "complications")
- duration_days: Duration in days (integer, null if not specified)
- body_part: Affected body part/location (string, null if not specified)
- age_bracket: Patient age bracket (e.g., "30-40", null if not specified)
- gender: Patient gender (if mentioned)
- severity: Condition severity (e.g., "mild", "moderate", "severe", null if not specified)

Clinical note:
{clinical_text}

Return ONLY valid JSON, no other text."""

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,
                        response_mime_type="application/json"
                    ),
                    contents=prompt
                )

                # Parse JSON response
                entities = json.loads(response.text)

                # Validate required fields
                if not entities.get('condition') or not entities.get('treatment'):
                    logger.warning(f"Missing required fields in extraction: {entities}")
                    return None

                logger.debug(f"Extracted entities: {entities}")
                return entities

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}): {e}")
                continue

            except Exception as e:
                logger.error(f"Error extracting entities (attempt {attempt + 1}): {e}")
                continue

        logger.error(f"Failed to extract entities after {max_retries} attempts")
        return None

    def batch_extract(
        self,
        clinical_texts: List[str],
        skip_failures: bool = True
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Extract entities from multiple clinical texts

        Args:
            clinical_texts: List of clinical note texts
            skip_failures: Skip failed extractions instead of raising error

        Returns:
            List of extracted entity dictionaries
        """
        results = []

        for i, text in enumerate(clinical_texts):
            try:
                entities = self.extract_entities(text)

                if entities is None and not skip_failures:
                    raise ValueError(f"Failed to extract entities from text {i}")

                results.append(entities)

            except Exception as e:
                if skip_failures:
                    logger.warning(f"Skipping text {i} due to error: {e}")
                    results.append(None)
                else:
                    raise

        success_count = sum(1 for r in results if r is not None)
        logger.info(
            f"Batch extraction: {success_count}/{len(clinical_texts)} successful"
        )

        return results

    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of a clinical query

        Args:
            query: Query text

        Returns:
            Intent category (diagnosis, treatment, history, prognosis, general)
        """
        system_instruction = """You are a medical intent classifier.
Return ONLY the category name, nothing else."""

        prompt = f"""Classify the intent of this clinical query into ONE category:
- diagnosis: Identifying or confirming a medical condition
- treatment: Planning or recommending treatment
- history: Reviewing patient history or past cases
- prognosis: Predicting outcomes or recovery
- general: General medical information

Query: {query}

Return ONLY the category name, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                ),
                contents=prompt
            )

            intent = response.text.strip().lower()

            # Validate intent
            valid_intents = {'diagnosis', 'treatment', 'history', 'prognosis', 'general'}
            if intent not in valid_intents:
                logger.warning(f"Invalid intent: {intent}, defaulting to 'general'")
                return 'general'

            return intent

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return 'general'

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptoms from clinical text

        Args:
            text: Clinical note text

        Returns:
            List of symptoms
        """
        system_instruction = """You are a medical symptom extractor.
Return ONLY a valid JSON array of symptoms."""

        prompt = f"""Extract all symptoms mentioned in this clinical note.
Return a JSON array of symptom strings.

Clinical note:
{text}

Return ONLY a valid JSON array, no other text."""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json"
                ),
                contents=prompt
            )

            symptoms = json.loads(response.text)

            if not isinstance(symptoms, list):
                logger.warning(f"Invalid symptoms format: {symptoms}")
                return []

            return symptoms

        except Exception as e:
            logger.error(f"Error extracting symptoms: {e}")
            return []

    def generate_insight_description(
        self,
        condition: str,
        treatment: str,
        statistics: Dict[str, Any]
    ) -> str:
        """
        Generate natural language description of aggregated insight

        Args:
            condition: Medical condition
            treatment: Treatment/procedure
            statistics: Aggregated statistics

        Returns:
            Natural language description
        """
        system_instruction = """You are a medical insights generator.
Generate concise, professional medical descriptions suitable for healthcare providers."""

        prompt = f"""Generate a concise, professional medical insight description (2-3 sentences) from this data:

Condition: {condition}
Treatment: {treatment}
Statistics: {json.dumps(statistics, indent=2)}

Focus on:
1. Treatment effectiveness (outcome distribution)
2. Typical duration/recovery time
3. Sample size and evidence strength

Use professional medical language suitable for healthcare providers."""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=200
                ),
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Treatment of {condition} with {treatment} based on {statistics.get('sample_size', 0)} cases."


def main():
    """CLI entry point for testing extraction"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract medical entities from text")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Clinical text to extract entities from"
    )

    args = parser.parse_args()

    extractor = MedicalEntityExtractor()
    entities = extractor.extract_entities(args.text)

    print(json.dumps(entities, indent=2))


if __name__ == "__main__":
    main()
