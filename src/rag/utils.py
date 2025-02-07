import tiktoken
import re

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens 

def rmSpace(text: str) -> str:
    """Remove extra spaces from text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip() 