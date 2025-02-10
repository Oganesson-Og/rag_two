from typing import Dict, List, Optional, Union, Tuple
import re
import sympy
from latex2sympy2 import latex2sympy
from sympy import latex
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

class MathDomain(Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    LINEAR_ALGEBRA = "linear_algebra"
    NUMBER_THEORY = "number_theory"

@dataclass
class MathExpression:
    """Represents a mathematical expression with context."""
    original_text: str
    latex_form: str
    sympy_form: Optional[sympy.Expr]
    variables: List[str]
    domain: MathDomain
    complexity_score: float
    prerequisites: List[str]
    metadata: Dict

class MathContentProcessor:
    """Processor for mathematical content in educational materials."""
    
    def __init__(
        self,
        math_patterns_path: Optional[Path] = None,
        complexity_threshold: float = 0.7
    ):
        self.logger = logging.getLogger(__name__)
        self.complexity_threshold = complexity_threshold
        
        # Load mathematical patterns and rules
        self.math_patterns = self._load_math_patterns(math_patterns_path)
        
        # Initialize symbol mappings
        self.symbol_mappings = {
            '∫': 'integral',
            '∑': 'sum',
            '∏': 'product',
            '∂': 'partial',
            '∇': 'nabla',
            '∆': 'delta',
            '≈': 'approx',
            '≠': 'neq',
            '≤': 'leq',
            '≥': 'geq'
        }
        
        # Initialize domain-specific keywords
        self.domain_keywords = self._initialize_domain_keywords()

    def _load_math_patterns(self, patterns_path: Optional[Path]) -> Dict:
        """Load mathematical patterns and rules from configuration."""
        default_patterns = {
            'equation_markers': [r'\begin{equation}', r'\[', r'\(', '$'],
            'special_functions': ['\\sin', '\\cos', '\\tan', '\\log', '\\exp'],
            'operators': ['+', '-', '*', '/', '^', '='],
            'grouping': ['(', ')', '[', ']', '{', '}'],
        }
        
        if patterns_path and patterns_path.exists():
            try:
                with open(patterns_path, 'r') as f:
                    custom_patterns = json.load(f)
                default_patterns.update(custom_patterns)
            except Exception as e:
                self.logger.error(f"Error loading math patterns: {str(e)}")
                
        return default_patterns

    def _initialize_domain_keywords(self) -> Dict[MathDomain, List[str]]:
        """Initialize keywords for different mathematical domains."""
        return {
            MathDomain.ALGEBRA: [
                'polynomial', 'equation', 'variable', 'factor', 'root'
            ],
            MathDomain.CALCULUS: [
                'derivative', 'integral', 'limit', 'differential', 'series'
            ],
            MathDomain.GEOMETRY: [
                'angle', 'triangle', 'circle', 'polygon', 'vector'
            ],
            MathDomain.STATISTICS: [
                'probability', 'distribution', 'mean', 'variance', 'correlation'
            ],
            MathDomain.LINEAR_ALGEBRA: [
                'matrix', 'vector', 'eigenvalue', 'determinant', 'linear'
            ],
            MathDomain.NUMBER_THEORY: [
                'prime', 'factor', 'divisor', 'modulo', 'congruence'
            ]
        }

    def process_content(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, List[MathExpression]]:
        """Process content containing mathematical expressions."""
        # Extract mathematical expressions
        math_expressions = self._extract_math_expressions(content)
        
        # Process each expression
        processed_expressions = []
        processed_content = content
        
        for expr in math_expressions:
            try:
                processed_expr = self._process_expression(expr)
                processed_expressions.append(processed_expr)
                
                # Update content with processed expression
                processed_content = self._update_content(
                    processed_content,
                    expr,
                    processed_expr
                )
            except Exception as e:
                self.logger.error(f"Error processing expression: {str(e)}")
                
        return processed_content, processed_expressions

    def _extract_math_expressions(self, content: str) -> List[str]:
        """Extract mathematical expressions from content."""
        expressions = []
        
        # Extract delimited math expressions
        for marker in self.math_patterns['equation_markers']:
            pattern = f"{marker}(.*?){marker}"
            matches = re.finditer(pattern, content, re.DOTALL)
            expressions.extend([m.group(1) for m in matches])
            
        # Extract inline math expressions
        inline_pattern = r'\$([^$]+)\$'
        inline_matches = re.finditer(inline_pattern, content)
        expressions.extend([m.group(1) for m in inline_matches])
        
        return expressions

    def _process_expression(self, expr: str) -> MathExpression:
        """Process a single mathematical expression."""
        # Clean and normalize expression
        cleaned_expr = self._clean_expression(expr)
        
        try:
            # Convert to SymPy
            sympy_expr = latex2sympy(cleaned_expr)
            
            # Extract variables
            variables = list(sympy_expr.free_symbols)
            
            # Determine mathematical domain
            domain = self._determine_domain(cleaned_expr, str(sympy_expr))
            
            # Calculate complexity
            complexity = self._calculate_complexity(sympy_expr)
            
            # Extract prerequisites
            prerequisites = self._extract_prerequisites(sympy_expr)
            
            return MathExpression(
                original_text=expr,
                latex_form=cleaned_expr,
                sympy_form=sympy_expr,
                variables=[str(v) for v in variables],
                domain=domain,
                complexity_score=complexity,
                prerequisites=prerequisites,
                metadata={}
            )
        except Exception as e:
            self.logger.warning(f"Failed to process expression: {str(e)}")
            return MathExpression(
                original_text=expr,
                latex_form=cleaned_expr,
                sympy_form=None,
                variables=[],
                domain=MathDomain.ALGEBRA,  # default
                complexity_score=0.0,
                prerequisites=[],
                metadata={'error': str(e)}
            )

    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize mathematical expression."""
        # Remove unnecessary whitespace
        expr = re.sub(r'\s+', ' ', expr.strip())
        
        # Normalize symbols
        for symbol, replacement in self.symbol_mappings.items():
            expr = expr.replace(symbol, f'\\{replacement}')
            
        return expr

    def _determine_domain(self, latex_expr: str, sympy_expr: str) -> MathDomain:
        """Determine the mathematical domain of an expression."""
        domain_scores = {domain: 0 for domain in MathDomain}
        
        # Check keywords presence
        for domain, keywords in self.domain_keywords.items():
            score = sum(
                1 for keyword in keywords
                if keyword in latex_expr.lower() or keyword in sympy_expr.lower()
            )
            domain_scores[domain] = score
            
        # Return domain with highest score
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    def _calculate_complexity(self, expr: sympy.Expr) -> float:
        """Calculate complexity score for an expression."""
        try:
            # Factors contributing to complexity:
            # 1. Number of operations
            # 2. Depth of expression tree
            # 3. Number of special functions
            # 4. Number of variables
            
            count = 0
            depth = 0
            
            def traverse(e, level=0):
                nonlocal count, depth
                count += 1
                depth = max(depth, level)
                for arg in e.args:
                    traverse(arg, level + 1)
            
            traverse(expr)
            
            # Normalize complexity score between 0 and 1
            complexity = (count * 0.4 + depth * 0.6) / 20
            return min(max(complexity, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 0.5

    def _extract_prerequisites(self, expr: sympy.Expr) -> List[str]:
        """Extract prerequisite concepts for understanding the expression."""
        prerequisites = set()
        
        # Add basic prerequisites based on expression structure
        str_expr = str(expr)
        
        if 'derivative' in str_expr or 'diff' in str_expr:
            prerequisites.add('derivatives')
        if 'integral' in str_expr:
            prerequisites.add('integration')
        if 'matrix' in str_expr:
            prerequisites.add('matrix_operations')
        if 'sin' in str_expr or 'cos' in str_expr:
            prerequisites.add('trigonometry')
        if '^' in str_expr:
            prerequisites.add('exponents')
            
        return list(prerequisites)

    def _update_content(
        self,
        content: str,
        original_expr: str,
        processed_expr: MathExpression
    ) -> str:
        """Update content with processed mathematical expression."""
        # If processing failed, return original content
        if not processed_expr.sympy_form:
            return content
            
        # Replace original expression with processed version
        return content.replace(
            original_expr,
            f"${processed_expr.latex_form}$"
        ) 