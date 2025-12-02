"""Document model for tracking uploaded documents."""

from datetime import datetime
from . import db


class Document(db.Model):
    """Model for tracking uploaded and indexed documents."""

    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=True)
    file_size = db.Column(db.Integer, nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    mime_type = db.Column(db.String(100), nullable=True)

    # Content info
    character_count = db.Column(db.Integer, default=0)
    word_count = db.Column(db.Integer, default=0)
    chunk_count = db.Column(db.Integer, default=0)

    # Metadata
    title = db.Column(db.String(255), nullable=True)
    category = db.Column(db.String(100), nullable=False, default="general", index=True)
    department = db.Column(db.String(100), nullable=False, default="all", index=True)
    tags = db.Column(db.JSON, default=list)
    description = db.Column(db.Text, nullable=True)

    # Indexing status
    is_indexed = db.Column(db.Boolean, default=False)
    indexed_at = db.Column(db.DateTime, nullable=True)
    indexing_error = db.Column(db.Text, nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Foreign keys
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    def mark_indexed(self, chunk_count):
        """Mark document as successfully indexed."""
        self.is_indexed = True
        self.indexed_at = datetime.utcnow()
        self.chunk_count = chunk_count
        self.indexing_error = None

    def mark_indexing_failed(self, error):
        """Mark document as failed to index."""
        self.is_indexed = False
        self.indexing_error = str(error)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "title": self.title,
            "category": self.category,
            "department": self.department,
            "tags": self.tags,
            "description": self.description,
            "is_indexed": self.is_indexed,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "indexing_error": self.indexing_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "uploaded_by": self.uploaded_by,
        }

    def to_summary(self):
        """Return a summary for listing."""
        return {
            "source_id": self.source_id,
            "title": self.title or self.original_filename,
            "category": self.category,
            "department": self.department,
            "chunk_count": self.chunk_count,
            "is_indexed": self.is_indexed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Document {self.source_id}>"
