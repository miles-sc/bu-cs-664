"""
Test script for InputAssessor class.
Tests the assessor with various customer input messages and pretty prints the results.
"""

# Suppress gRPC and Google Cloud warnings
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import json
from input_assessor import InputAssessor


def pretty_print_assessment(input_message: str, assessment: dict) -> None:
    """
    Pretty print the assessment results.

    Args:
        input_message: The original customer input
        assessment: The assessment dictionary from the API
    """
    print("\n" + "="*80)
    print(f"INPUT: {input_message}")
    print("="*80)

    # Group by category
    emotions = {}
    faqs = {}
    other = {}

    for key, value in assessment.items():
        match_score, confidence = value
        if key.startswith("Input expresses"):
            emotions[key] = (match_score, confidence)
        elif key.startswith("FAQ:"):
            faqs[key] = (match_score, confidence)
        else:
            other[key] = (match_score, confidence)

    # Print emotions
    if emotions:
        print("\nEMOTIONS:")
        for emotion, (score, conf) in emotions.items():
            if score > 0.0:  # Only show non-zero scores
                emotion_name = emotion.replace("Input expresses ", "")
                print(f"  {emotion_name:15} - Match: {score:4.1f}/10.0  Confidence: {conf:4.1f}/10.0")

    # Print FAQs
    if faqs:
        print("\nRELEVANT FAQs:")
        for faq, (score, conf) in faqs.items():
            if score > 0.0:  # Only show non-zero scores
                faq_name = faq.replace("FAQ: ", "")
                print(f"  {faq_name}")
                print(f"    Match: {score:4.1f}/10.0  Confidence: {conf:4.1f}/10.0")

    # Print other assessments
    if other:
        print("\nOTHER ASSESSMENTS:")
        for key, (score, conf) in other.items():
            if score > 0.0:  # Only show non-zero scores
                print(f"  {key}")
                print(f"    Match: {score:4.1f}/10.0  Confidence: {conf:4.1f}/10.0")

    print()


def main():
    """Run tests with various customer inputs."""

    print("Initializing InputAssessor...")
    assessor = InputAssessor()

    # Test cases with different customer scenarios
    test_inputs = [
        "I'm so frustrated! My phone won't turn on and I have an important meeting in 10 minutes!",
        "Can someone please help me? I need to speak to a real person right now.",
        "How do I set up my new phone? I just got it and I'm not sure where to start.",
        "I love this phone! It's the best one I've ever had. Thank you!",
        "I forgot my password and can't access my phone. What should I do?",
        "My phone keeps showing error code 404. How do I fix this?",
        "Where can I get my phone repaired? The screen is cracked.",
    ]

    print(f"\nRunning assessments on {len(test_inputs)} test inputs...\n")

    for i, input_message in enumerate(test_inputs, 1):
        try:
            print(f"\n[Test {i}/{len(test_inputs)}]")
            assessment = assessor.assess(input_message)
            pretty_print_assessment(input_message, assessment)

        except Exception as e:
            print(f"\n[Test {i}/{len(test_inputs)}]")
            print(f"ERROR processing input: {input_message}")
            print(f"Error: {e}\n")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
