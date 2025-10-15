"""
Natlang chatbot class.
A natural language chatbot agent for handling customer service interactions.
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress gRPC/ALTS warnings before importing genai
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

from input_assessor import InputAssessor
import google.generativeai as genai


class Natlang:
    """
    Natural language chatbot agent that uses InputAssessor to analyze
    and respond to customer messages.
    """

    # Thresholds for decision making
    MEDIUM_SCORE = 5.0
    MEDIUM_CONFIDENCE = 5.0

    # URLs and resources
    FAQ_URLS = {
        "FAQ: How to activate and set up your phone": "https://example.com/faq/setup",
        "FAQ: How to use advanced features and custom settings": "https://example.com/faq/advanced-features",
        "FAQ: How to troubleshoot error messages and common issues": "https://example.com/faq/troubleshooting",
        "FAQ: How to find certified repair locations and get your phone repaired": "https://example.com/faq/repair-locations",
        "FAQ: How to back up and restore data on your phone": "https://example.com/faq/backup-restore",
        "FAQ: How to reset your phone's password": "https://example.com/faq/password-reset"
    }
    REVIEW_URL = "https://example.com/reviews/earbuds"
    FEEDBACK_URL = "https://example.com/feedback"
    RETURN_LABEL_URL = "https://example.com/downloads/return_label.pdf"

    def __init__(self):
        """
        Initialize the Natlang chatbot with an InputAssessor instance.
        """
        self.assessor = InputAssessor()
        self.human_request_counter = 0
        self.awaiting_return_info = False
        self.awaiting_faq_feedback = False
        self.routed_to_human = False

        # Initialize Gemini model for response generation
        env_path = Path(__file__).parent / '.env'
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.response_model = genai.GenerativeModel('gemini-2.5-flash')

    def get_assessment(self, user_input: str) -> dict:
        """
        Assess a user's input message and return the assessment scores.

        Args:
            user_input: The customer's message to analyze

        Returns:
            dict: A dictionary containing assessment scores for each parameter.
                  Each value is a list of two floats: [match_score, confidence].

        Raises:
            ValueError: If the API response is not valid JSON
            Exception: If the assessment fails
        """
        return self.assessor.assess(user_input)

    def generate_response(self, user_input: str, instructions: str) -> str:
        """
        Generate a response using Gemini API based on the user input and instructions.

        Args:
            user_input: The customer's message
            instructions: Instructions for how to respond to the customer

        Returns:
            str: The generated response from Gemini
        """
        prompt = f"""{instructions}

Customer message: "{user_input}"

Generate an appropriate response:"""

        response = self.response_model.generate_content(prompt)
        return response.text.strip()

    def route_to_human(self, user_input: str, reason: str = "general inquiry") -> str:
        """
        Route customer to a live agent with appropriate message.

        Args:
            user_input: The customer's message
            reason: The reason for routing (e.g., "customer anger", "customer request",
                    "faqs didn't help", "return info issue", "general inquiry")

        Returns:
            str: Response notifying customer they're being routed to an agent
        """
        self.routed_to_human = True

        instructions = f"""You are a customer service chatbot. You need to route the customer to a live agent.

Reason for routing: {reason}

Generate a brief, empathetic message that:
1. Acknowledges the situation based on the reason provided
2. Notifies them you are connecting them with a live agent who can better assist them
3. Thanks them for their patience
Keep the tone professional, helpful, and respectful."""

        return self.generate_response(user_input, instructions)

    def validate_return_info(self, user_input: str) -> bool:
        """
        Check if user input contains a 5-digit zip code and 6-character order number.

        Args:
            user_input: The customer's message

        Returns:
            bool: True if the input contains both required pieces of information in correct format
        """
        prompt = """You are a validator for customer service information. Analyze the following customer message to determine if it contains BOTH:
1. A 5-digit zip code (numbers only, exactly 5 digits)
2. A 6-character order number (alphanumeric, exactly 6 characters)

Customer message: "{}"

Respond with ONLY "YES" if BOTH pieces of information are present in the correct format, or "NO" if either is missing or in the wrong format.""".format(user_input)

        response = self.response_model.generate_content(prompt)
        result = response.text.strip().upper()
        return result == "YES"

    def return_processing(self, user_input: str, assessment: dict) -> str:
        """
        Handle return request with apology and required information collection.

        Args:
            user_input: The customer's message
            assessment: The assessment dictionary (unused, kept for compatibility)

        Returns:
            str: Response handling the return request
        """
        instructions = """You are a customer service chatbot. The customer is requesting a return.
Generate a response that:
1. Apologizes for any issues and provides assurance
2. Requests their 5-digit zip code and 6-character order number to process the return
3. Mentions that once we have this information, we'll send them a return shipping label to mail back their product
Keep the tone professional, helpful, and empathetic."""

        self.awaiting_return_info = True
        return self.generate_response(user_input, instructions)

    def handle_return_info_response(self, user_input: str) -> str:
        """
        Handle the customer's response after requesting return information.
        Validates if they provided the correct format (5-digit zip + 6-character order number).

        Args:
            user_input: The customer's message containing return information

        Returns:
            str: Response either confirming return processing or routing to human
        """
        if self.validate_return_info(user_input):
            # Valid format provided
            instructions = f"""You are a customer service chatbot. The customer has provided their zip code and order number for a return.
Generate a response that:
1. Thanks them for providing the information
2. Confirms that they will receive a refund as soon as their return has been processed
3. Provides them with this link to download their return shipping label: {self.RETURN_LABEL_URL}
Keep the tone professional, helpful, and grateful."""

            self.awaiting_return_info = False
            return self.generate_response(user_input, instructions)
        else:
            # Invalid format - route to human
            self.awaiting_return_info = False
            return self.route_to_human(user_input, "invalid return information format")

    def provide_faqs(self, user_input: str, assessment: dict) -> str:
        """
        Provide relevant FAQ links based on assessment scores.

        Args:
            user_input: The customer's message
            assessment: The assessment dictionary with FAQ scores

        Returns:
            str: Response with FAQ links and follow-up question
        """
        # Find all FAQ matches with good scores
        relevant_faqs = []
        for faq_key, url in self.FAQ_URLS.items():
            if faq_key in assessment:
                score, confidence = assessment[faq_key]
                if score >= self.MEDIUM_SCORE and confidence >= self.MEDIUM_CONFIDENCE:
                    relevant_faqs.append((faq_key, url))

        if not relevant_faqs:
            # No FAQs matched, shouldn't normally happen if this method is called
            return self.route_to_human(user_input, "no matching FAQ articles found")

        # Build FAQ list for the instructions
        faq_list = "\n".join([f"- {faq.replace('FAQ: ', '')}: {url}" for faq, url in relevant_faqs])

        instructions = f"""You are a customer service chatbot. The customer's question matches one or more FAQ articles.
Generate a response that:
1. Acknowledges their question in a respectful and helpful tone
2. Provides the following FAQ link(s) and recommends they read the article(s):
{faq_list}
3. Asks if the document(s) helped with their issue
Keep the tone professional, helpful, and respectful."""

        self.awaiting_faq_feedback = True
        return self.generate_response(user_input, instructions)

    def validate_faq_response(self, user_input: str) -> bool:
        """
        Check if the customer indicates the FAQ documents helped them.

        Args:
            user_input: The customer's message

        Returns:
            bool: True if customer indicates they were helped, False if not
        """
        prompt = """You are a validator for customer service interactions. Analyze the following customer message to determine if the customer is indicating that the FAQ documentation helped them resolve their issue.

Customer message: "{}"

Look for positive indicators like:
- "Yes", "Yeah", "That helped", "That worked", "Thanks, I'm good now"
- Expressions of gratitude or satisfaction
- Confirmation that their problem is solved

Look for negative indicators like:
- "No", "Nope", "That didn't help", "Still having issues", "I'm still confused"
- Requests for more help
- Expressions of frustration

Respond with ONLY "YES" if the customer indicates they were helped, or "NO" if they indicate they were not helped or need more assistance.""".format(user_input)

        response = self.response_model.generate_content(prompt)
        result = response.text.strip().upper()
        return result == "YES"

    def handle_faq_feedback_response(self, user_input: str) -> str:
        """
        Handle the customer's response after providing FAQ documents.
        Determines if the FAQs helped and either continues conversation or routes to human.

        Args:
            user_input: The customer's message about whether FAQs helped

        Returns:
            str: Response either asking if they need more help or routing to human
        """
        if self.validate_faq_response(user_input):
            # FAQs helped - ask if there's anything else
            instructions = """You are a customer service chatbot. The customer has indicated that the FAQ documents helped them.
Generate a response that:
1. Expresses happiness that the documentation was helpful
2. Asks if there is anything else you can help them with
Keep the tone friendly, professional, and helpful."""

            self.awaiting_faq_feedback = False
            return self.generate_response(user_input, instructions)
        else:
            # FAQs didn't help - route to human
            self.awaiting_faq_feedback = False
            return self.route_to_human(user_input, "FAQ documents didn't help")

    def handle_positive_feedback(self, user_input: str) -> str:
        """
        Handle positive feedback by thanking customer and requesting a review.

        Args:
            user_input: The customer's message

        Returns:
            str: Response thanking customer and requesting review
        """
        instructions = f"""You are a customer service chatbot. The customer is happy and providing positive feedback about their product.
Generate a response that:
1. Thanks the customer warmly for their positive feedback
2. Expresses appreciation for their satisfaction with the product
3. Asks them to consider sharing their experience by leaving a public review at: {self.REVIEW_URL}
Keep the tone friendly, appreciative, and professional."""

        return self.generate_response(user_input, instructions)

    def process_input(self, user_input: str) -> str:
        """
        Main decision tree for processing customer input and generating responses.

        Args:
            user_input: The customer's message

        Returns:
            str: The chatbot's response based on the decision tree
        """
        # Check if we're awaiting return information
        if self.awaiting_return_info:
            return self.handle_return_info_response(user_input)

        # Check if we're awaiting FAQ feedback
        if self.awaiting_faq_feedback:
            return self.handle_faq_feedback_response(user_input)

        # Get assessment for the input
        assessment = self.get_assessment(user_input)

        # Extract key scores
        anger_score, anger_conf = assessment.get("Input expresses anger", [0.0, 0.0])
        human_request_score, human_request_conf = assessment.get(
            "Input contains a request to contact a live agent or human", [0.0, 0.0]
        )
        happy_score, happy_conf = assessment.get("Input expresses happiness", [0.0, 0.0])
        positive_feedback_score, positive_feedback_conf = assessment.get(
            "Input expresses positive feedback about their phone", [0.0, 0.0]
        )
        return_request_score, return_request_conf = assessment.get(
            "Input contains a request for a refund or return", [0.0, 0.0]
        )

        # Decision Tree (in priority order)

        # 1. Check for high anger - route to human immediately
        if anger_score >= self.MEDIUM_SCORE and anger_conf >= self.MEDIUM_CONFIDENCE:
            return self.route_to_human(user_input, reason="customer expressing anger or frustration")

        # 2. Check for human request
        if human_request_score >= self.MEDIUM_SCORE and human_request_conf >= self.MEDIUM_CONFIDENCE:
            if self.human_request_counter == 0:
                # First request - ask them to describe their issue
                self.human_request_counter = 1
                instructions = """You are a customer service chatbot. The customer has requested to speak with a human agent.
Generate a response that:
1. Acknowledges their request
2. Asks them to first describe their issue so you can try to help or route them to the right department
3. Reassures them that you can connect them with an agent if needed
Keep the tone helpful and respectful."""
                return self.generate_response(user_input, instructions)
            else:
                # Second request - route to human
                return self.route_to_human(user_input, reason="customer repeatedly requested human agent")

        # 3. Check for return request
        if return_request_score >= self.MEDIUM_SCORE and return_request_conf >= self.MEDIUM_CONFIDENCE:
            return self.return_processing(user_input, assessment)

        # 4. Check for FAQ matches
        has_faq_match = False
        for faq_key in self.FAQ_URLS.keys():
            if faq_key in assessment:
                score, confidence = assessment[faq_key]
                if score >= self.MEDIUM_SCORE and confidence >= self.MEDIUM_CONFIDENCE:
                    has_faq_match = True
                    break
        if has_faq_match:
            return self.provide_faqs(user_input, assessment)

        # 5. Check for positive feedback
        if (happy_score >= self.MEDIUM_SCORE and happy_conf >= self.MEDIUM_CONFIDENCE and
            positive_feedback_score >= self.MEDIUM_SCORE and positive_feedback_conf >= self.MEDIUM_CONFIDENCE):
            return self.handle_positive_feedback(user_input)

        # 6. Default - route to human
        return self.route_to_human(user_input, reason="inquiry outside chatbot capabilities")
