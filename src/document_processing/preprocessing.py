import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from .models import PreprocessingConfig

class TextPreprocessor:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def clean_text(self, text: str, config: Optional[PreprocessingConfig] = None, **kwargs) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
            
        config = config or PreprocessingConfig(**kwargs)
        
        # Apply cleaning steps based on config
        if config.remove_html:
            text = BeautifulSoup(text, 'html.parser').get_text()
            
        if config.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            
        if config.normalize_unicode:
            text = self._normalize_unicode(text)
            
        # Normalize whitespace
        text = ' '.join(text.split())
        
        if config.case == 'lower':
            text = text.lower()
        elif config.case == 'upper':
            text = text.upper()
            
        return text
        
    def clean_texts(self, texts: List[str], **kwargs) -> List[str]:
        return [self.clean_text(text, **kwargs) for text in texts]
        
    def _normalize_unicode(self, text: str) -> str:
        import unicodedata
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8') 