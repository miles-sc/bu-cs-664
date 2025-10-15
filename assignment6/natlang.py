"""
Natlang chatbot class.
A natural language chatbot agent for handling customer service interactions.
"""

from input_assessor import InputAssessor
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv


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

    def __init__(self):
        """
        Initialize the Natlang chatbot with an InputAssessor instance.
        """
        self.assessor = InputAssessor()
        self.human_request_counter = 0

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

    def route_to_human(self, user_input: str, reason: str = "general") -> str:
        """
        Route customer to a live agent with appropriate message.

        Args:
            user_input: The customer's message
            reason: The reason for routing (e.g., "anger", "request", "general")

        Returns:
            str: Response notifying customer they're being routed to an agent
        """
        if reason == "anger":
            instructions = """You are a customer service chatbot. The customer appears angry or frustrated.
Generate a brief, empathetic message that:
1. Acknowledges their frustration
2. Notifies them you are routing them to a live agent immediately
3. Thanks them for their patience
Keep the tone professional, calm, and respectful."""
        else:
            instructions = """You are a customer service chatbot. You need to route the customer to a live agent.
Generate a brief, helpful message that:
1. Notifies them you are connecting them with a live agent
2. Thanks them for their patience
Keep the tone professional and helpful."""

        return self.generate_response(user_input, instructions)

    def refund_processing(self, user_input: str, assessment: dict) -> str:
        """
        Handle refund request with apology and required information collection.

        Args:
            user_input: The customer's message
            assessment: The assessment dictionary to check for issues mentioned

        Returns:
            str: Response handling the refund request
        """
        # Check if customer mentioned any issues (sadness, anger, disgust, problems)
        has_issue = False
        issue_indicators = [
            "Input expresses sadness",
            "Input expresses anger",
            "Input expresses disgust",
            "FAQ: How to troubleshoot error messages and common issues"
        ]
        for indicator in issue_indicators:
            if indicator in assessment:
                score, confidence = assessment[indicator]
                if score >= self.MEDIUM_SCORE and confidence >= self.MEDIUM_CONFIDENCE:
                    has_issue = True
                    break

        instructions = f"""You are a customer service chatbot. The customer is requesting a refund.
Generate a response that:
1. {"Apologizes for any issues they experienced with the product" if has_issue else "Acknowledges their refund request"}
2. Requests their zip code and order number to process the refund
3. Mentions that once we have this information, we'll send them a return shipping label to mail back their product
4. Offers a link to provide private feedback on the product: {self.FEEDBACK_URL}
Keep the tone professional, helpful, and empathetic."""

        return self.generate_response(user_input, instructions)

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
            return self.route_to_human(user_input, "general")

        # Build FAQ list for the instructions
        faq_list = "\n".join([f"- {faq.replace('FAQ: ', '')}: {url}" for faq, url in relevant_faqs])

        instructions = f"""You are a customer service chatbot. The customer's question matches one or more FAQ articles.
Generate a response that:
1. Acknowledges their question in a respectful and helpful tone
2. Provides the following FAQ link(s) and recommends they read the article(s):
{faq_list}
3. Asks if the document(s) helped with their issue
4. Mentions that if the FAQ doesn't help, you can connect them with a live agent
Keep the tone professional, helpful, and respectful."""

        return self.generate_response(user_input, instructions)

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
        refund_request_score, refund_request_conf = assessment.get(
            "Input contains a request for a refund or return", [0.0, 0.0]
        )

        # Decision Tree (in priority order)

        # 1. Check for high anger - route to human immediately
        if anger_score >= self.MEDIUM_SCORE and anger_conf >= self.MEDIUM_CONFIDENCE:
            return self.route_to_human(user_input, reason="anger")

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
                return self.route_to_human(user_input, reason="request")

        # 3. Check for refund request
        if refund_request_score >= self.MEDIUM_SCORE and refund_request_conf >= self.MEDIUM_CONFIDENCE:
            return self.refund_processing(user_input, assessment)

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
        return self.route_to_human(user_input, reason="general")
