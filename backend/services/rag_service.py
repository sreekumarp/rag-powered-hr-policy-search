"""Enhanced RAG service with query logging and analytics."""

import uuid
import time
from datetime import datetime
from flask import request, g
from models import db, QueryLog
from utils.rag_pipeline import RAGPipeline


class RAGService:
    """Service for RAG operations with analytics and logging."""

    def __init__(self, vector_store, llm_service=None, cache=None, config=None):
        """
        Initialize RAG service.

        Args:
            vector_store: Vector store instance
            llm_service: LLM service instance (optional)
            cache: Response cache instance (optional)
            config: Configuration dict
        """
        config = config or {}

        self.pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_service=llm_service,
            cache=cache,
            retrieval_k=config.get("retrieval_k", 3),
            confidence_threshold=config.get("confidence_threshold", 0.3),
            min_confidence_for_answer=config.get("min_confidence_for_answer", 0.5),
            enable_llm=config.get("enable_llm", True),
            enable_citations=config.get("enable_citations", True),
        )

    def query(self, question, filters=None, include_low_confidence=False, user_id=None):
        """
        Execute a RAG query with full logging.

        Args:
            question: User question
            filters: Optional metadata filters
            include_low_confidence: Include low confidence results
            user_id: ID of querying user (optional)

        Returns:
            Query result dict with answer, sources, and metadata
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        status = "success"
        error_message = None

        try:
            # Execute query through pipeline
            result = self.pipeline.query(question, filters, include_low_confidence)

            # Determine status based on confidence
            if result.get("confidence", 0) < 0.5:
                status = "low_confidence"

            # Add query ID to result
            result["query_id"] = query_id

            return result

        except Exception as e:
            status = "error"
            error_message = str(e)
            raise

        finally:
            # Log query
            total_time = (time.time() - start_time) * 1000
            self._log_query(
                query_id=query_id,
                question=question,
                filters=filters,
                result=result if "result" in locals() else None,
                latency_ms=total_time,
                status=status,
                error_message=error_message,
                user_id=user_id,
            )

    def _log_query(
        self,
        query_id,
        question,
        filters,
        result,
        latency_ms,
        status,
        error_message,
        user_id,
    ):
        """Log query to database."""
        try:
            log_entry = QueryLog(
                query_id=query_id,
                question=question,
                question_length=len(question),
                filters_applied=filters or {},
                answer=result.get("answer") if result else None,
                confidence_score=result.get("confidence") if result else None,
                sources_retrieved=result.get("relevant_chunks", 0) if result else 0,
                sources_used=[s.get("source") for s in result.get("sources", [])] if result else [],
                latency_ms=latency_ms,
                status=status,
                error_message=error_message,
                warnings=result.get("warnings", []) if result else [],
                user_id=user_id,
                ip_address=request.remote_addr if request else None,
                user_agent=request.headers.get("User-Agent", "")[:512] if request else None,
            )

            db.session.add(log_entry)
            db.session.commit()

        except Exception as e:
            # Don't fail the query if logging fails
            db.session.rollback()
            print(f"Failed to log query: {e}")

    def get_query_history(self, user_id=None, limit=50, offset=0):
        """
        Get query history with optional user filtering.

        Args:
            user_id: Filter by user ID
            limit: Max results
            offset: Pagination offset

        Returns:
            List of query logs
        """
        query = QueryLog.query

        if user_id:
            query = query.filter_by(user_id=user_id)

        logs = (
            query.order_by(QueryLog.created_at.desc()).offset(offset).limit(limit).all()
        )

        return [log.to_dict() for log in logs]

    def get_analytics(self, days=30):
        """
        Get query analytics for the specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Analytics dict
        """
        stats = QueryLog.get_statistics(days)

        # Add pipeline info
        pipeline_info = self.pipeline.get_pipeline_info()

        return {
            "query_statistics": stats,
            "pipeline_configuration": pipeline_info.get("configuration", {}),
            "vector_store_stats": pipeline_info.get("vector_store", {}),
        }

    def get_popular_queries(self, limit=10):
        """Get most common queries (simplified version)."""
        from sqlalchemy import func

        # Get queries by similar patterns (simple word count for now)
        recent_queries = (
            QueryLog.query.order_by(QueryLog.created_at.desc()).limit(100).all()
        )

        # Group by first few words (simple approach)
        query_patterns = {}
        for log in recent_queries:
            words = log.question.lower().split()[:5]
            pattern = " ".join(words)
            if pattern not in query_patterns:
                query_patterns[pattern] = {
                    "pattern": log.question,
                    "count": 0,
                    "avg_confidence": 0,
                    "total_confidence": 0,
                }
            query_patterns[pattern]["count"] += 1
            if log.confidence_score:
                query_patterns[pattern]["total_confidence"] += log.confidence_score

        # Calculate averages and sort
        patterns = []
        for pattern, data in query_patterns.items():
            if data["count"] > 0:
                data["avg_confidence"] = round(
                    data["total_confidence"] / data["count"], 4
                )
            del data["total_confidence"]
            patterns.append(data)

        patterns.sort(key=lambda x: x["count"], reverse=True)
        return patterns[:limit]

    def get_low_confidence_queries(self, threshold=0.5, limit=20):
        """Get queries with low confidence scores for improvement."""
        logs = (
            QueryLog.query.filter(QueryLog.confidence_score < threshold)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "query_id": log.query_id,
                "question": log.question,
                "confidence_score": log.confidence_score,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]

    def index_sample_documents(self, faq_path):
        """Index sample FAQ documents."""
        return self.pipeline.add_sample_documents(faq_path)

    def reset(self):
        """Reset pipeline statistics."""
        self.pipeline.reset_statistics()

    def get_pipeline_info(self):
        """Get pipeline configuration and stats."""
        return self.pipeline.get_pipeline_info()
