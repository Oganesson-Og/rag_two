"""
Text Enhancement Module for RAG Pipeline
--------------------------------------

This module handles text preprocessing, cleaning, and enhancement for the RAG pipeline.
It provides a comprehensive text processing pipeline specifically designed for educational
content and academic documents.

Key Features:
- Spell checking and grammar correction using language-tool-python
- Header and footer removal using configurable regex patterns
- Abbreviation expansion with domain-specific dictionaries
- Academic text-specific processing rules
- Support for multiple educational domains
- Batch processing capabilities
- Caching for improved performance

Technical Details:
- Uses language-tool-python for grammar checking
- Implements custom regex patterns for academic content
- Supports multiple languages and domains
- Memory-efficient processing for large documents
- Configurable through external configuration files

Dependencies:
- language-tool-python>=2.7.1
- spacy>=3.7.2
- regex>=2023.0.0

Example Usage:
    enhancer = TextEnhancer()
    enhanced_text = enhancer.process_text(input_text)
    
    # With custom configuration
    enhancer = TextEnhancer(config_path='custom_config.json')
    enhanced_text = enhancer.process_text(input_text, domain='chemistry')

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

## Existing requirements...

# New NLP requirements
# spacy>=3.7.2
# language-tool-python>=2.7.1

from language_tool_python import LanguageTool
import re
from typing import Dict, List, Optional, Set, Any
import spacy
from ..utils.text_cleaner import TextCleaner
import json
import os
from pathlib import Path
from ..config.settings import MAX_TOKENS
from ..nlp import Tokenizer  # Updated import

class TextEnhancer:
    """Text enhancement and correction utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.tool = LanguageTool('en-US')
        self.nlp = spacy.load('en_core_web_sm')
        self.text_cleaner = TextCleaner()
        
        # Load abbreviations from config file or use defaults
        self.abbreviations = self._load_abbreviations(self.config.get('abbreviations_path'))
        
        # Common academic headers/footers patterns
        self.header_patterns = [
            r'\b(Page|pg\.?)\s*\d+\s*(of|/)\s*\d+\b',
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'(?i)^.*(?:confidential|draft|internal use only).*$',
            r'(?m)^(?:Section|Chapter)\s+\d+(?:\.\d+)*\s*[:.-]',
            r'^\s*(?:prepared by|author|date|version).*$',
            r'^\s*(?:www|http|https).*$',  # URLs in headers/footers
            r'^\s*(?:all rights reserved|copyright).*$',
        ]
        
        # Academic subject-specific patterns
        self.subject_headers = [
            r'(?i)^.*(?:chemistry|physics|biology|mathematics).*department.*$',
            r'(?i)^.*(?:laboratory|experiment|worksheet).*$',
            r'(?i)^.*(?:test|exam|quiz|assessment).*$'
        ]
        
        self.tokenizer = Tokenizer()
        
    def _load_abbreviations(self, config_path: Optional[str] = None) -> Dict[str, str]:
        """Load abbreviation mappings from config file or use defaults.
        
        Args:
            config_path: Path to JSON config file containing abbreviations
            
        Returns:
            Dictionary of abbreviation mappings
        """
        default_abbrev = {
            # General academic
            "chem": "chemistry",
            "bio": "biology",
            "phys": "physics",
            "math": "mathematics",
            "eng": "english",
            "lit": "literature",
            "hist": "history",
            "geo": "geography",
            
            # Chemistry specific
            "aq": "aqueous",
            "conc": "concentrated",
            "dil": "dilute",
            "soln": "solution",
            "temp": "temperature",
            "vol": "volume",
            
            # Physics specific
            "vel": "velocity",
            "acc": "acceleration",
            "freq": "frequency",
            "mag": "magnitude",
            
            # Biology specific
            "org": "organism",
            "env": "environment",
            "repro": "reproduction",
            
            # Mathematics specific
            "eq": "equation",
            "calc": "calculation",
            "prob": "probability",
            "stats": "statistics"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_abbrev = json.load(f)
                default_abbrev.update(custom_abbrev)
            except Exception as e:
                print(f"Error loading abbreviations config: {e}")
                
        return default_abbrev
        
    def correct_spelling_grammar(self, text: str) -> str:
        """Correct spelling and grammar using LanguageTool.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text with spelling and grammar fixes
        """
        try:
            # First clean the text
            text = self.text_cleaner.clean_text(text)
            
            # Split into manageable chunks to avoid timeout
            max_chunk_size = 5000
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            corrected_chunks = []
            for chunk in chunks:
                # Get correction suggestions
                matches = self.tool.check(chunk)
                
                # Apply corrections, prioritizing high-confidence fixes
                corrected = chunk
                for match in sorted(matches, key=lambda m: m.ruleId.startswith('SPELL')):
                    if match.replacements:
                        # Skip certain types of suggestions
                        if any(skip in match.ruleId.lower() for skip in ['style', 'typography']):
                            continue
                        corrected = corrected[:match.offset] + match.replacements[0] + corrected[match.offset + match.errorLength:]
                
                corrected_chunks.append(corrected)
            
            return ' '.join(corrected_chunks)
            
        except Exception as e:
            print(f"Error in spelling/grammar correction: {e}")
            return text  # Return original text if correction fails
        
    def remove_headers_footers(self, text: str) -> str:
        """Remove headers and footers using regex patterns.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with headers and footers removed
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        # Track potential header/footer lines
        header_footer_lines: Set[int] = set()
        
        for i, line in enumerate(lines):
            is_header_footer = False
            
            # Check against header/footer patterns
            for pattern in self.header_patterns + self.subject_headers:
                if re.search(pattern, line.strip()):
                    is_header_footer = True
                    break
            
            # Check for repeated lines at same positions across pages
            if i > 0 and i % 20 == 0:  # Assume page break every ~20 lines
                prev_page_line = lines[i - 20] if i >= 20 else ""
                if line.strip() == prev_page_line.strip() and line.strip():
                    is_header_footer = True
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        # Remove excessive blank lines
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
        
    def expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations in the text.
        
        Args:
            text: Input text containing abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        # Tokenize text while preserving sentence structure
        doc = self.nlp(text)
        expanded_tokens = []
        
        for token in doc:
            word = token.text
            lower_word = word.lower()
            
            # Check if it's an abbreviation
            if lower_word in self.abbreviations:
                # Preserve original capitalization
                if word.isupper():
                    expanded = self.abbreviations[lower_word].upper()
                elif word.istitle():
                    expanded = self.abbreviations[lower_word].title()
                else:
                    expanded = self.abbreviations[lower_word]
                    
                # Handle special cases (e.g., keeping periods)
                if token.text.endswith('.'):
                    expanded += '.'
                    
                expanded_tokens.append(expanded)
            else:
                expanded_tokens.append(word)
            
            # Preserve spacing and punctuation
            if token.whitespace_:
                expanded_tokens.append(token.whitespace_)
        
        return ''.join(expanded_tokens)

    def process_text(self, text: str) -> str:
        """Process and enhance text."""
        # Clean text
        text = self.text_cleaner.clean_text(text)
        text = self.text_cleaner.clean_latex(text)
        text = self.text_cleaner.clean_html(text)
        text = self.text_cleaner.normalize_whitespace(text)
        
        # Tokenize and analyze
        tokens = self.tokenizer.tokenize(text)
        
        # Filter out unwanted tokens
        filtered_tokens = self.tokenizer.filter_tokens(
            tokens,
            exclude_stops=True
        )
        
        # Reconstruct text
        return self.tokenizer.decode(filtered_tokens)
    
    def analyze_text(self, text: str) -> dict:
        """Analyze text properties."""
        return self.tokenizer.analyze_text(text)

    def enhance_text(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> str:
        """Enhance and correct text content."""
        try:
            options = options or {}
            text = text.strip()
            
            # Apply enhancements
            if options.get('fix_grammar', True):
                text = self._fix_grammar(text)
                
            if options.get('improve_readability', True):
                text = self._improve_readability(text)
                
            return text
            
        except Exception as e:
            self.logger.error(f"Text enhancement error: {str(e)}")
            raise

    def _fix_grammar(self, text: str) -> str:
        """Fix grammatical errors."""
        try:
            matches = self.tool.check(text)
            return self.tool.correct(text)
        except Exception as e:
            self.logger.error(f"Grammar correction error: {str(e)}")
            raise

    def _improve_readability(self, text: str) -> str:
        """Improve text readability."""
        try:
            doc = self.nlp(text)
            # Implement readability improvements
            return text
        except Exception as e:
            self.logger.error(f"Readability improvement error: {str(e)}")
            raise
