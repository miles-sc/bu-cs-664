import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai


class InputAssessor:
    """
    A class that uses Google Gemini API to assess customer messages
    against predefined parameters and return structured JSON output.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the InputAssessor with Google Gemini API credentials.

        Args:
            api_key: Google Gemini API key (optional - will load from .env if not provided)
        """
        # Load environment variables from .env file
        if api_key is None:
            env_path = Path(__file__).parent / '.env'
            load_dotenv(dotenv_path=env_path)
            api_key = os.getenv('GEMINI_API_KEY')

            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")

        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        self.system_prompt = """You are an assistant that analyzes a customer's message to a cell phone company chatbot.
Your task is to output a single, valid JSON object that assesses the input message against several predefined parameters.

Instructions:

    1) For each parameter below, output a tuple of two floats:

        - The first value = how strongly the input matches the parameter (0.0 = not at all, 10.0 = perfectly matches).
        - The second value = your confidence in that assessment (0.0 = no confidence, 10.0 = full confidence).

    2) Always include all parameters, even if the values are (0.0, 0.0).

    3) Output only valid JSON — no explanations, extra text, or commentary.

        - The JSON must begin with { and end with }.
        - All keys must be quoted strings.
        - All float values must include one decimal place (e.g., 0.0, 7.5).
        - Do not include trailing commas

    4) The top-level JSON structure must exactly follow this format:
{
  "Input expresses happiness": [0.0, 0.0],
  "Input expresses sadness": [0.0, 0.0],
  "Input expresses fear": [0.0, 0.0],
  "Input expresses anger": [0.0, 0.0],
  "Input expresses disgust": [0.0, 0.0],
  "Input expresses surprise": [0.0, 0.0],
  "FAQ: How to activate and set up your phone": [0.0, 0.0],
  "FAQ: How to use advanced features and custom settings": [0.0, 0.0],
  "FAQ: How to troubleshoot error messages and common issues": [0.0, 0.0],
  "FAQ: How to find certified repair locations and get your phone repaired": [0.0, 0.0],
  "FAQ: How to back up and restore data on your phone": [0.0, 0.0],
  "FAQ: How to reset your phone's password": [0.0, 0.0],
  "Input expresses positive feedback about their phone": [0.0, 0.0],
  "Input contains a request to contact a live agent or human": [0.0, 0.0],
  "Input contains a request for a refund or return": [0.0, 0.0]
}

(End instructions)

Follow the above instructions in evaluation of the following customer input message:
"""

    def assess(self, input_message: str) -> dict:
        """
        Assess a customer message using Google Gemini API.

        Args:
            input_message: The customer's message to analyze

        Returns:
            dict: A dictionary containing assessment scores for each parameter.
                  Each value is a list of two floats: [match_score, confidence].

        Raises:
            ValueError: If the API response is not valid JSON
            Exception: If the API call fails
        """
        # Construct the full prompt
        full_prompt = f'{self.system_prompt}\n"{input_message}"'

        try:
            # Query the Gemini API
            response = self.model.generate_content(full_prompt)

            # Extract the text response
            response_text = response.text.strip()

            # Strip markdown code fences if present (e.g., ```json ... ```)
            # Remove opening code fence
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'^```\s*', '', response_text)
            # Remove closing code fence
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()

            # Parse the JSON response
            assessment_result = json.loads(response_text)

            # Validate the format
            self._validate_format(assessment_result)

            return assessment_result

        except json.JSONDecodeError as e:
            raise ValueError(f"API response is not valid JSON: {e}\nResponse: {response_text}")
        except Exception as e:
            raise Exception(f"Error querying Gemini API: {e}")

    def _validate_format(self, result: dict) -> None:
        """
        Validate that the API response matches the expected format.

        Args:
            result: The parsed JSON result from the API

        Raises:
            ValueError: If the format is invalid
        """
        expected_keys = [
            "Input expresses happiness",
            "Input expresses sadness",
            "Input expresses fear",
            "Input expresses anger",
            "Input expresses disgust",
            "Input expresses surprise",
            "FAQ: How to activate and set up your phone",
            "FAQ: How to use advanced features and custom settings",
            "FAQ: How to troubleshoot error messages and common issues",
            "FAQ: How to find certified repair locations and get your phone repaired",
            "FAQ: How to back up and restore data on your phone",
            "FAQ: How to reset your phone's password",
            "Input expresses positive feedback about their phone",
            "Input contains a request to contact a live agent or human",
            "Input contains a request for a refund or return"
        ]

        # Check all expected keys are present
        for key in expected_keys:
            if key not in result:
                raise ValueError(f"Missing expected key: {key}")

            # Check that each value is a list of two numbers
            value = result[key]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Invalid value for '{key}': expected list of 2 floats, got {value}")

            if not all(isinstance(v, (int, float)) for v in value):
                raise ValueError(f"Invalid value for '{key}': values must be numeric, got {value}")
