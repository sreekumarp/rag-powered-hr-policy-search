"""Query log model for analytics and audit trail."""

from datetime import datetime
from . import db


class QueryLog(db.Model):
    """Model for logging all queries for analytics and auditing."""

    __tablename__ = "query_logs"

    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.String(36), unique=True, nullable=False, index=True)

    # Query details
    question = db.Column(db.Text, nullable=False)
    question_length = db.Column(db.Integer, nullable=False)
    filters_applied = db.Column(db.JSON, default=dict)

    # Response details
    answer = db.Column(db.Text, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    sources_retrieved = db.Column(db.Integer, default=0)
    sources_used = db.Column(db.JSON, default=list)

    # Performance metrics
    latency_ms = db.Column(db.Float, nullable=False)
    embedding_time_ms = db.Column(db.Float, nullable=True)
    retrieval_time_ms = db.Column(db.Float, nullable=True)
    generation_time_ms = db.Column(db.Float, nullable=True)

    # Status
    status = db.Column(db.String(20), nullable=False, default="success")  # success, error, low_confidence
    error_message = db.Column(db.Text, nullable=True)
    warnings = db.Column(db.JSON, default=list)

    # User context
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "question_length": self.question_length,
            "filters_applied": self.filters_applied,
            "answer": self.answer,
            "confidence_score": self.confidence_score,
            "sources_retrieved": self.sources_retrieved,
            "sources_used": self.sources_used,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_id": self.user_id,
        }

    def to_analytics(self):
        """Return analytics-focused summary."""
        return {
            "query_id": self.query_id,
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "sources_retrieved": self.sources_retrieved,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def get_statistics(days=30):
        """Get query statistics for the last N days."""
        from datetime import timedelta
        from sqlalchemy import func

        cutoff = datetime.utcnow() - timedelta(days=days)

        stats = db.session.query(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
            func.avg(QueryLog.confidence_score).label("avg_confidence"),
            func.min(QueryLog.latency_ms).label("min_latency_ms"),
            func.max(QueryLog.latency_ms).label("max_latency_ms"),
        ).filter(QueryLog.created_at >= cutoff).first()

        status_counts = (
            db.session.query(QueryLog.status, func.count(QueryLog.id))
            .filter(QueryLog.created_at >= cutoff)
            .group_by(QueryLog.status)
            .all()
        )

        return {
            "period_days": days,
            "total_queries": stats.total_queries or 0,
            "avg_latency_ms": round(stats.avg_latency_ms or 0, 2),
            "avg_confidence": round(stats.avg_confidence or 0, 4),
            "min_latency_ms": round(stats.min_latency_ms or 0, 2),
            "max_latency_ms": round(stats.max_latency_ms or 0, 2),
            "status_breakdown": {status: count for status, count in status_counts},
        }

    def __repr__(self):
        return f"<QueryLog {self.query_id}>"
