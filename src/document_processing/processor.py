from .diagram_analyzer import DiagramAnalyzer

def is_diagram(page):
    """Check if a page contains a diagram.
    
    Args:
        page: The page to check
        
    Returns:
        bool: True if page contains a diagram, False otherwise
    """
    # Add your diagram detection logic here
    # For example, check for image elements, diagram keywords, etc.
    return hasattr(page, 'image') or 'diagram' in str(page).lower()

class DocumentProcessor:
    def __init__(self):
        self.diagram_analyzer = DiagramAnalyzer()
    
    def process_page(self, page):
        # When a diagram is detected
        if is_diagram(page):
            diagram_data = self.diagram_analyzer.analyze_diagram(page)