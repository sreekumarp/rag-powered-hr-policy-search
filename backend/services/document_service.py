"""Document service for managing uploads and indexing."""

import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from models import db, Document
from utils.document_processor import DocumentProcessor


class DocumentService:
    """Service for document management operations."""

    def __init__(self, upload_folder, vector_store, allowed_extensions=None):
        """
        Initialize document service.

        Args:
            upload_folder: Directory for uploaded files
            vector_store: Vector store instance for indexing
            allowed_extensions: Set of allowed file extensions
        """
        self.upload_folder = upload_folder
        self.vector_store = vector_store
        self.allowed_extensions = allowed_extensions or {
            ".pdf",
            ".docx",
            ".doc",
            ".txt",
            ".html",
            ".md",
        }
        self.processor = DocumentProcessor(upload_folder)

        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

    def validate_file(self, file):
        """
        Validate uploaded file.

        Returns:
            (valid, error_message)
        """
        if not file:
            return False, "No file provided"

        if file.filename == "":
            return False, "No file selected"

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"Unsupported file type: {ext}"

        return True, None

    def upload_and_index(self, file, user_id=None, metadata=None):
        """
        Upload a document and index it in the vector store.

        Args:
            file: File object from request
            user_id: ID of uploading user (optional)
            metadata: Additional metadata dict

        Returns:
            (success, result) tuple
        """
        metadata = metadata or {}

        # Validate file
        valid, error = self.validate_file(file)
        if not valid:
            return False, {"error": error}

        try:
            # Save file
            file_path = self.processor.save_uploaded_file(file, file.filename)

            # Process document (extract text)
            doc_info = self.processor.process_file(file_path)

            # Generate source ID
            source_id = f"{metadata.get('title', doc_info['filename'])}_{doc_info['file_hash'][:8]}"

            # Check for duplicate
            existing = Document.query.filter_by(file_hash=doc_info["file_hash"]).first()
            if existing:
                return False, {
                    "error": "Document already exists",
                    "existing_source_id": existing.source_id,
                }

            # Create database record
            document = Document(
                source_id=source_id,
                filename=secure_filename(file.filename),
                original_filename=file.filename,
                file_path=file_path,
                file_size=doc_info["file_size"],
                file_hash=doc_info["file_hash"],
                mime_type=doc_info.get("mime_type"),
                character_count=doc_info["char_count"],
                word_count=doc_info["word_count"],
                title=metadata.get("title", doc_info["filename"]),
                category=metadata.get("category", "general"),
                department=metadata.get("department", "all"),
                tags=metadata.get("tags", []),
                description=metadata.get("description"),
                uploaded_by=user_id,
            )

            db.session.add(document)

            # Index in vector store
            vector_metadata = {
                "category": document.category,
                "department": document.department,
                "title": document.title,
                "file_type": os.path.splitext(file.filename)[1].lower(),
                "document_id": document.id,
            }

            result = self.vector_store.add_document(
                doc_info["extracted_text"], source_id, vector_metadata
            )

            # Update document with indexing info
            document.mark_indexed(result["chunks_added"])
            db.session.commit()

            return True, {
                "document": document.to_dict(),
                "chunks_added": result["chunks_added"],
                "total_chunks": result["total_chunks"],
            }

        except Exception as e:
            db.session.rollback()
            # Try to clean up the uploaded file
            if "file_path" in locals():
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            return False, {"error": str(e)}

    def get_document(self, source_id):
        """Get document by source ID."""
        document = Document.query.filter_by(source_id=source_id).first()
        return document.to_dict() if document else None

    def list_documents(self, category=None, department=None, sort_by="created_at", limit=100):
        """
        List documents with optional filtering.

        Args:
            category: Filter by category
            department: Filter by department
            sort_by: Sort field (created_at, title, chunk_count)
            limit: Maximum number of results

        Returns:
            List of document summaries
        """
        query = Document.query

        if category:
            query = query.filter_by(category=category)
        if department:
            query = query.filter_by(department=department)

        # Apply sorting
        if sort_by == "title":
            query = query.order_by(Document.title.asc())
        elif sort_by == "chunk_count":
            query = query.order_by(Document.chunk_count.desc())
        else:  # created_at
            query = query.order_by(Document.created_at.desc())

        documents = query.limit(limit).all()
        return [doc.to_summary() for doc in documents]

    def delete_document(self, source_id, user_id=None):
        """
        Delete a document and remove from vector store.

        Args:
            source_id: Document source ID
            user_id: ID of user performing deletion (for audit)

        Returns:
            (success, result) tuple
        """
        document = Document.query.filter_by(source_id=source_id).first()
        if not document:
            return False, {"error": "Document not found"}

        try:
            # Remove from vector store
            self.vector_store.delete_document(source_id)

            # Delete file from disk
            if document.file_path and os.path.exists(document.file_path):
                os.remove(document.file_path)

            # Delete database record
            db.session.delete(document)
            db.session.commit()

            return True, {"message": f"Document {source_id} deleted successfully"}

        except Exception as e:
            db.session.rollback()
            return False, {"error": str(e)}

    def get_statistics(self):
        """Get document statistics."""
        from sqlalchemy import func

        stats = db.session.query(
            func.count(Document.id).label("total_documents"),
            func.sum(Document.chunk_count).label("total_chunks"),
            func.sum(Document.character_count).label("total_characters"),
            func.sum(Document.word_count).label("total_words"),
            func.sum(Document.file_size).label("total_file_size"),
        ).first()

        category_counts = (
            db.session.query(Document.category, func.count(Document.id))
            .group_by(Document.category)
            .all()
        )

        return {
            "total_documents": stats.total_documents or 0,
            "total_chunks": int(stats.total_chunks or 0),
            "total_characters": int(stats.total_characters or 0),
            "total_words": int(stats.total_words or 0),
            "total_file_size_bytes": int(stats.total_file_size or 0),
            "categories": {cat: count for cat, count in category_counts},
        }

    def update_metadata(self, source_id, metadata):
        """
        Update document metadata.

        Args:
            source_id: Document source ID
            metadata: Dict of fields to update

        Returns:
            (success, result) tuple
        """
        document = Document.query.filter_by(source_id=source_id).first()
        if not document:
            return False, {"error": "Document not found"}

        # Update allowed fields
        allowed_fields = ["title", "category", "department", "tags", "description"]
        for field in allowed_fields:
            if field in metadata:
                setattr(document, field, metadata[field])

        db.session.commit()
        return True, document.to_dict()
