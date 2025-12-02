"""
Document Processing Module

Handles parsing of various document formats:
- PDF files
- DOCX files
- Plain text files
- HTML files
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class DocumentProcessor:
    """
    Process various document formats and extract text content.
    """

    SUPPORTED_FORMATS = {".pdf", ".docx", ".doc", ".txt", ".html", ".md"}

    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize document processor.

        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def process_file(self, file_path: str) -> Dict:
        """
        Process a document file and extract text.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing extracted text and metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")

        # Extract text based on file type
        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext in {".docx", ".doc"}:
            text = self._extract_docx(file_path)
        elif ext in {".txt", ".md"}:
            text = self._extract_text(file_path)
        elif ext == ".html":
            text = self._extract_html(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(file_path)

        # Build metadata
        metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "file_size": path.stat().st_size,
            "file_hash": file_hash,
            "file_type": ext,
            "extracted_text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "processed_at": datetime.utcnow().isoformat(),
        }

        return metadata

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2

            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF processing. Install with: pip install PyPDF2"
            )

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            import docx2txt

            return docx2txt.process(file_path)
        except ImportError:
            raise ImportError(
                "docx2txt is required for DOCX processing. Install with: pip install docx2txt"
            )

    def _extract_text(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback to basic text extraction
            return self._extract_text(file_path)

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def save_uploaded_file(self, file_storage, filename: str) -> str:
        """
        Save an uploaded file to disk.

        Args:
            file_storage: Flask FileStorage object
            filename: Desired filename

        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)

        # Add timestamp to avoid collisions
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_filename)
        unique_filename = f"{name}_{timestamp}{ext}"

        file_path = os.path.join(self.upload_dir, unique_filename)

        # Save file
        file_storage.save(file_path)

        return file_path

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        # Remove path components
        filename = os.path.basename(filename)
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        # Remove special characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        filename = "".join(c if c in safe_chars else "_" for c in filename)
        return filename

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.SUPPORTED_FORMATS)

    def validate_file_size(self, file_size: int, max_size_mb: int = 50) -> bool:
        """
        Validate file size.

        Args:
            file_size: File size in bytes
            max_size_mb: Maximum allowed size in MB

        Returns:
            True if valid, False otherwise
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
