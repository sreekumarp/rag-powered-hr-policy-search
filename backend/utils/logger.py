"""
Logging Configuration Module

Provides structured logging for the HR Assistant application.
"""

import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry)


def setup_logger(
    name: str = "hr_assistant",
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup application logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)

    # File handler (JSON structured)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    # Error file handler (errors only)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def log_with_data(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Log message with additional structured data.

    Args:
        logger: Logger instance
        level: Log level (info, debug, warning, error)
        message: Log message
        **kwargs: Additional data to include
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "",
        0,
        message,
        (),
        None,
    )
    record.extra_data = kwargs
    logger.handle(record)


# Create default logger
logger = setup_logger()


# Convenience functions
def log_query(question: str, confidence: float, sources: int, latency_ms: float):
    """Log a query event."""
    log_with_data(
        logger,
        "info",
        f"Query processed: {question[:50]}...",
        event="query",
        question_length=len(question),
        confidence=confidence,
        sources_retrieved=sources,
        latency_ms=latency_ms,
    )


def log_document_indexed(filename: str, chunks: int, processing_time_ms: float):
    """Log document indexing event."""
    log_with_data(
        logger,
        "info",
        f"Document indexed: {filename}",
        event="index",
        filename=filename,
        chunks_created=chunks,
        processing_time_ms=processing_time_ms,
    )


def log_error(error_type: str, message: str, **kwargs):
    """Log an error event."""
    log_with_data(
        logger,
        "error",
        f"{error_type}: {message}",
        event="error",
        error_type=error_type,
        **kwargs,
    )


def log_api_request(method: str, endpoint: str, status_code: int, latency_ms: float):
    """Log API request."""
    log_with_data(
        logger,
        "info",
        f"{method} {endpoint} - {status_code}",
        event="api_request",
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        latency_ms=latency_ms,
    )
