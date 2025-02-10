from dependency_injector import containers, providers
from .src.document_processing.processors.diagram_archive import DiagramAnalyzer

class Container(containers.DeclarativeContainer):
    diagram_analyzer = providers.Singleton(DiagramAnalyzer)