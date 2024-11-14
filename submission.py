import json
import collections
import argparse
import random
import numpy as np
import requests
import re

# api key for query. see https://docs.together.ai/docs/get-started
def your_api_key():
    YOUR_API_KEY = '35aa8e364eb9f6f2b4c6e9eb61a7f124be16db13e4d5c7335d9cde39620fe0ab'
    return YOUR_API_KEY


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    prefix = '''[Question: what is 1234567+1234567?]\n[Answer: 2469134]\n
            [Question: what is 234567+234567?]\n[Answer: 469134]\n
            [Question: what is 567890+123456?]\n[Answer: 691346]\n
            [Question: what is '''

    suffix = '?]\n[Answer: '

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys. 
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 80,  # max_tokens must be >= 50 because we don't always have prior on output length
        'temperature': 0.2,  # Lowered to reduce randomness and make results more deterministic
        'top_k': 30,         # Reduced to ensure the model focuses on the top probable choices
        'top_p': 0.5,        # Slightly reduced to narrow down the sampling set
        'repetition_penalty': 1.1,  # Increased to penalize repetitive outputs and encourage correct results
        'stop': []
    }
    
    return config


def your_pre_processing(s):
    return s.strip()

    
def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    first_line = output_string.splitlines()[0]  # Only consider the first line for the answer
    only_digits = re.sub(r"\D", "", first_line)  # Extract digits only
    try:
        res = int(only_digits)
    except ValueError:
        res = 0  # Return 0 if no valid number is found
    return res