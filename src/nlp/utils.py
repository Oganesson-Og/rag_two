def get_project_base_directory():
    """Returns the base directory of the project"""
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    return False

def word_tokenize(text):
    # Implement word tokenization logic here
    # Or import from nltk: from nltk import word_tokenize
    return text.split() 