"""
Token Counter Module
------------------

Token counting utilities using tiktoken.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import tiktoken
from typing import Optional

def num_tokens_from_string(
    string: str,
    encoding_name: str = "cl100k_base"
) -> int:
    """
    Returns the number of tokens in a text string.
    
    Args:
        string: The text to count tokens for
        encoding_name: The name of the tiktoken encoding to use
        
    Returns:
        int: Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens 