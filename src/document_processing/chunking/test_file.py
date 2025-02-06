import spacy
import language_tool_python

# Test spaCy
nlp = spacy.load('en_core_web_sm')
doc = nlp("This is a test sentence.")
print("spaCy test:", [token.text for token in doc])

# Test language-tool-python
tool = language_tool_python.LanguageTool('en-US')
text = "This is a test sentense."  # intentional misspelling
matches = tool.check(text)
print("Language tool test:", matches)