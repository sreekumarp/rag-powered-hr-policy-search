"""
Enhanced RAG Pipeline for HR Assistant - Phase 2

Improvements:
- Metadata filtering support
- Confidence validation
- Better response generation
- Query analytics
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
from utils.vector_store import VectorStore
from utils.logger import log_query, log_error, logger


class RAGPipeline:
    """Enhanced RAG pipeline with filtering and validation."""

    def __init__(
        self,
        vector_store: VectorStore,
        retrieval_k: int = 3,
        confidence_threshold: float = 0.3,
        min_confidence_for_answer: float = 0.5,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: VectorStore instance
            retrieval_k: Number of documents to retrieve
            confidence_threshold: Minimum score to consider relevant
            min_confidence_for_answer: Minimum confidence to provide answer
        """
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.confidence_threshold = confidence_threshold
        self.min_confidence_for_answer = min_confidence_for_answer

        # Query statistics
        self.query_count = 0
        self.total_latency_ms = 0
        self.average_confidence = 0

    def query(
        self,
        question: str,
        filters: Dict = None,
        include_low_confidence: bool = False,
    ) -> Dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            question: User's question
            filters: Optional metadata filters (category, department, etc.)
            include_low_confidence: Return results even if confidence is low

        Returns:
            Dictionary with answer, sources, confidence, and metadata
        """
        start_time = time.time()

        # Track query
        self.query_count += 1

        # Retrieve documents with optional filtering
        try:
            retrieved_docs = self.vector_store.search(
                question, k=self.retrieval_k, filter_metadata=filters
            )
        except Exception as e:
            log_error("RetrievalError", str(e), question=question[:50])
            return self._error_response(
                "Failed to retrieve documents", question, start_time
            )

        if not retrieved_docs:
            return self._empty_response(question, start_time, filters)

        # Filter by confidence threshold
        relevant_docs = [
            (doc, score)
            for doc, score in retrieved_docs
            if score >= self.confidence_threshold
        ]

        if not relevant_docs:
            return self._low_confidence_response(
                question, start_time, retrieved_docs, include_low_confidence
            )

        # Build context and generate response
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(relevant_docs):
            context_parts.append(f"[Source {i+1}]: {doc['text']}")
            sources.append(
                {
                    "text": doc["text"][:300] + "..."
                    if len(doc["text"]) > 300
                    else doc["text"],
                    "source": doc.get("source", "unknown"),
                    "chunk_id": doc.get("chunk_id", 0),
                    "relevance_score": round(score, 4),
                    "category": doc.get("category", "general"),
                    "created_at": doc.get("created_at", ""),
                }
            )

        context = "\n\n".join(context_parts)

        # Generate response
        answer = self._generate_response(question, context, relevant_docs)

        # Calculate confidence
        avg_confidence = sum(score for _, score in relevant_docs) / len(relevant_docs)

        # Validate confidence
        warnings = []
        if avg_confidence < self.min_confidence_for_answer:
            warnings.append(
                f"Low confidence ({avg_confidence:.2f}). Answer may not be accurate."
            )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        self.total_latency_ms += latency_ms
        self.average_confidence = (
            (self.average_confidence * (self.query_count - 1) + avg_confidence)
            / self.query_count
        )

        # Log query
        log_query(question, avg_confidence, len(relevant_docs), latency_ms)

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": round(avg_confidence, 4),
            "relevant_chunks": len(relevant_docs),
            "query": question,
            "filters_applied": filters or {},
            "processing_time_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "warnings": warnings,
        }

        return result

    def _generate_response(
        self, question: str, context: str, docs: List
    ) -> str:
        """Generate a response based on retrieved context."""
        best_doc, best_score = docs[0]
        text = best_doc["text"]

        # Try to extract Q&A pairs
        if "Q:" in text and "A:" in text:
            qa_pairs = text.split("Q:")
            for qa in qa_pairs:
                if "A:" in qa:
                    parts = qa.split("A:", 1)
                    if len(parts) == 2:
                        q_part = parts[0].strip()
                        a_part = parts[1].strip().split("\n\n")[0]  # Get first paragraph
                        # Check relevance
                        if any(
                            word.lower() in q_part.lower()
                            for word in question.split()
                            if len(word) > 3
                        ):
                            return f"Based on HR policies:\n\n{a_part}"

        # Default response
        response = f"Based on the HR policies, here's relevant information:\n\n{text}"

        if len(docs) > 1:
            response += f"\n\n---\nFound {len(docs)} relevant sections. Check sources below for complete information."

        return response

    def _empty_response(
        self, question: str, start_time: float, filters: Dict = None
    ) -> Dict:
        """Response when no documents are found."""
        latency_ms = (time.time() - start_time) * 1000

        message = "I couldn't find any information to answer your question."
        if filters:
            message += f" Filters applied: {filters}. Try removing filters or asking a different question."

        log_query(question, 0.0, 0, latency_ms)

        return {
            "answer": message,
            "sources": [],
            "confidence": 0.0,
            "relevant_chunks": 0,
            "query": question,
            "filters_applied": filters or {},
            "processing_time_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "warnings": ["No relevant documents found"],
        }

    def _low_confidence_response(
        self,
        question: str,
        start_time: float,
        all_docs: List,
        include_anyway: bool,
    ) -> Dict:
        """Response when confidence is below threshold."""
        latency_ms = (time.time() - start_time) * 1000

        if include_anyway and all_docs:
            # Return best match anyway
            doc, score = all_docs[0]
            log_query(question, score, 1, latency_ms)

            return {
                "answer": f"Low confidence result:\n\n{doc['text'][:500]}...",
                "sources": [
                    {
                        "text": doc["text"][:300] + "...",
                        "source": doc.get("source", "unknown"),
                        "chunk_id": doc.get("chunk_id", 0),
                        "relevance_score": round(score, 4),
                    }
                ],
                "confidence": round(score, 4),
                "relevant_chunks": 1,
                "query": question,
                "processing_time_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat(),
                "warnings": [
                    "Low confidence result. May not be accurate.",
                    f"Best score: {score:.4f} (threshold: {self.confidence_threshold})",
                ],
            }

        max_score = max(score for _, score in all_docs) if all_docs else 0.0
        log_query(question, max_score, 0, latency_ms)

        return {
            "answer": "I couldn't find sufficiently relevant information. Try rephrasing your question or ask about a different topic.",
            "sources": [],
            "confidence": round(max_score, 4),
            "relevant_chunks": 0,
            "query": question,
            "processing_time_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "warnings": [
                f"Best match score ({max_score:.4f}) below threshold ({self.confidence_threshold})"
            ],
        }

    def _error_response(
        self, error_message: str, question: str, start_time: float
    ) -> Dict:
        """Response for errors."""
        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": f"An error occurred: {error_message}",
            "sources": [],
            "confidence": 0.0,
            "relevant_chunks": 0,
            "query": question,
            "processing_time_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "warnings": [f"Error: {error_message}"],
        }

    def add_sample_documents(self, faq_path: str = "data/sample_faq.txt") -> int:
        """Load sample FAQ documents with metadata."""
        try:
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_content = f.read()

            sections = faq_content.split("## ")
            total_chunks = 0

            for section in sections:
                if section.strip():
                    lines = section.strip().split("\n", 1)
                    if len(lines) >= 2:
                        title = lines[0].strip()
                        content = lines[1].strip()
                        source = f"HR_FAQ_{title.replace(' ', '_')}"

                        # Determine category from title
                        category = self._categorize_section(title)

                        metadata = {
                            "category": category,
                            "document_type": "faq",
                            "title": title,
                        }

                        result = self.vector_store.add_document(
                            content, source, metadata
                        )
                        total_chunks += result["chunks_added"]

            return total_chunks

        except FileNotFoundError:
            logger.error(f"FAQ file not found at {faq_path}")
            return 0
        except Exception as e:
            logger.error(f"Error loading FAQ: {e}")
            return 0

    def _categorize_section(self, title: str) -> str:
        """Categorize document section based on title."""
        title_lower = title.lower()

        if "leave" in title_lower or "vacation" in title_lower:
            return "leave_policies"
        elif "sick" in title_lower:
            return "sick_leave"
        elif "remote" in title_lower or "work from home" in title_lower:
            return "remote_work"
        elif "benefit" in title_lower or "compensation" in title_lower:
            return "benefits"
        elif "conduct" in title_lower or "harassment" in title_lower:
            return "workplace_conduct"
        elif "parental" in title_lower:
            return "parental_leave"
        elif "training" in title_lower or "development" in title_lower:
            return "training"
        else:
            return "general"

    def get_pipeline_info(self) -> Dict:
        """Get comprehensive pipeline information and statistics."""
        avg_latency = (
            self.total_latency_ms / self.query_count if self.query_count > 0 else 0
        )

        return {
            "configuration": {
                "retrieval_k": self.retrieval_k,
                "confidence_threshold": self.confidence_threshold,
                "min_confidence_for_answer": self.min_confidence_for_answer,
            },
            "statistics": {
                "total_queries": self.query_count,
                "average_latency_ms": round(avg_latency, 2),
                "average_confidence": round(self.average_confidence, 4),
            },
            "vector_store": self.vector_store.get_stats(),
        }

    def reset_statistics(self) -> None:
        """Reset query statistics."""
        self.query_count = 0
        self.total_latency_ms = 0
        self.average_confidence = 0
