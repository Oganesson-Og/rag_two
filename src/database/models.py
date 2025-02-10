"""
Database Models Module
------------------
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Document(Base):
    """Document model."""
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    content = Column(String, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    vectors = relationship('Vector', back_populates='document')

class Vector(Base):
    """Vector model."""
    __tablename__ = 'vectors'

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'))
    vector_data = Column(JSON, nullable=False)  # Store as JSON for flexibility
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship('Document', back_populates='vectors')

class Cache(Base):
    """Cache model."""
    __tablename__ = 'cache'

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)
    metadata = Column(JSON, default=dict)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 