# HR Assistant RAG - Complete Installation & Setup Guide

This guide provides step-by-step instructions to install, configure, and run all components of the HR Assistant RAG system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [System Requirements](#2-system-requirements)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Core Dependencies Installation](#4-core-dependencies-installation)
5. [Infrastructure Services Setup](#5-infrastructure-services-setup)
6. [Application Configuration](#6-application-configuration)
7. [Database Setup](#7-database-setup)
8. [Vector Store Configuration](#8-vector-store-configuration)
9. [Running the Application](#9-running-the-application)
10. [Monitoring & Observability Setup](#10-monitoring--observability-setup)
11. [Production Deployment](#11-production-deployment)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

### 1.1 Required Software

Before starting, ensure you have the following installed:

```bash
# Check versions
python --version    # Python 3.11+
docker --version    # Docker 24.0+
docker-compose --version  # Docker Compose 2.0+
git --version       # Git 2.0+
node --version      # Node.js 18+ (for frontend, optional)
```

### 1.2 Install Required Software

#### macOS (using Homebrew)

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install Docker Desktop
brew install --cask docker

# Install Git
brew install git

# Install additional tools
brew install postgresql@15  # For psql client
brew install redis          # For redis-cli
```

#### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Install Git
sudo apt install git -y

# Install PostgreSQL client
sudo apt install postgresql-client -y

# Install Redis tools
sudo apt install redis-tools -y
```

#### Windows (using WSL2)

```powershell
# Enable WSL2
wsl --install

# Install Ubuntu from Microsoft Store
# Then follow Ubuntu instructions above

# Or use Chocolatey
choco install python311
choco install docker-desktop
choco install git
```

### 1.3 API Keys Required

You'll need the following API keys:

- **OpenAI API Key**: For embeddings and LLM
  - Sign up at https://platform.openai.com/
  - Create API key at https://platform.openai.com/api-keys

- **Optional: Anthropic API Key**: For Claude fallback
  - Sign up at https://console.anthropic.com/

---

## 2. System Requirements

### 2.1 Minimum Requirements (Development)

| Component | Requirement |
|-----------|------------|
| CPU | 4 cores |
| RAM | 16 GB |
| Storage | 50 GB SSD |
| OS | macOS 12+, Ubuntu 20.04+, Windows 10+ (WSL2) |

### 2.2 Recommended Requirements (Production)

| Component | Requirement |
|-----------|------------|
| CPU | 8+ cores |
| RAM | 32 GB+ |
| Storage | 200 GB+ NVMe SSD |
| Network | 1 Gbps |

---

## 3. Development Environment Setup

### 3.1 Clone Repository

```bash
# Create project directory
mkdir -p ~/projects/hr-assistant
cd ~/projects/hr-assistant

# Clone repository (or initialize new)
git clone https://github.com/your-org/hr-assistant-rag.git .
# OR
git init
```

### 3.2 Create Project Structure

```bash
# Create directory structure
mkdir -p {app,workers,config,tests,scripts,docs,data}
mkdir -p app/{api,services,models,schemas,middleware}
mkdir -p app/api/routes
mkdir -p workers
mkdir -p config
mkdir -p tests/{unit,integration,e2e}
mkdir -p scripts/{migrations,seed}
mkdir -p data/{documents,indices,logs}

# Create initial files
touch app/__init__.py
touch app/main.py
touch app/api/__init__.py
touch app/services/__init__.py
touch app/models/__init__.py
touch requirements.txt
touch requirements-dev.txt
touch .env.example
touch .gitignore
touch docker-compose.yml
touch Dockerfile
touch README.md
```

### 3.3 Setup Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
.\venv\Scripts\activate

# Verify Python version
python --version  # Should show 3.11.x

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3.4 Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.*
!.env.example

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Data
data/documents/*
data/indices/*
data/logs/*
!data/documents/.gitkeep
!data/indices/.gitkeep
!data/logs/.gitkeep

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Docker
*.pid
*.seed
*.pid.lock
EOF

# Create .gitkeep files
touch data/documents/.gitkeep
touch data/indices/.gitkeep
touch data/logs/.gitkeep
```

---

## 4. Core Dependencies Installation

### 4.1 Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
psycopg2-binary==2.9.9

# Redis
redis==5.0.1
aioredis==2.0.1

# LangChain & LLM
langchain==0.0.350
langchain-openai==0.0.2
openai==1.3.7
tiktoken==0.5.2

# Vector Store
faiss-cpu==1.7.4
qdrant-client==1.6.9

# Document Processing
pypdf2==3.0.1
python-docx==1.1.0
unstructured==0.11.0
beautifulsoup4==4.12.2

# Data Validation
pydantic==2.5.2
pydantic-settings==2.1.0
email-validator==2.1.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
cryptography==41.0.7

# HTTP Client
httpx==0.25.2
aiohttp==3.9.1

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Utilities
python-dotenv==1.0.0
tenacity==8.2.3
numpy==1.26.2
boto3==1.33.6

# Object Storage
minio==7.2.0
EOF
```

### 4.2 Create requirements-dev.txt

```bash
cat > requirements-dev.txt << 'EOF'
-r requirements.txt

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.2
factory-boy==3.3.0

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.1
isort==5.12.0
bandit==1.7.6
safety==2.3.5

# Type Stubs
types-redis==4.6.0.11
types-requests==2.31.0.10

# Development Tools
ipython==8.17.2
ipdb==0.13.13
pre-commit==3.5.0
EOF
```

### 4.3 Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installations
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import faiss; print('FAISS: Installed')"
python -c "import qdrant_client; print(f'Qdrant: {qdrant_client.__version__}')"
```

### 4.4 Install Additional System Dependencies

```bash
# macOS
brew install poppler     # For PDF processing
brew install tesseract   # For OCR (optional)

# Ubuntu/Debian
sudo apt install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    pandoc
```

---

## 5. Infrastructure Services Setup

### 5.1 Create docker-compose.yml

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: hr_assistant_db
    environment:
      POSTGRES_DB: hr_assistant
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres_password_change_me
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - hr_network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: hr_assistant_redis
    command: redis-server --requirepass redis_password_change_me
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "redis_password_change_me", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - hr_network

  # Qdrant Vector Store
  qdrant:
    image: qdrant/qdrant:latest
    container_name: hr_assistant_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - hr_network

  # MinIO Object Storage (S3 Compatible)
  minio:
    image: minio/minio:latest
    container_name: hr_assistant_minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minio_admin
      MINIO_ROOT_PASSWORD: minio_password_change_me
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - hr_network

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    container_name: hr_assistant_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - hr_network

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    container_name: hr_assistant_grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: grafana_password_change_me
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - hr_network

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:
  prometheus_data:
  grafana_data:

networks:
  hr_network:
    driver: bridge
EOF
```

### 5.2 Create Prometheus Configuration

```bash
mkdir -p config

cat > config/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hr-assistant-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
```

### 5.3 Create Database Initialization Script

```bash
mkdir -p scripts

cat > scripts/init-db.sql << 'EOF'
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database if not exists (handled by Docker)
-- Additional setup can go here

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE hr_assistant TO postgres;
EOF
```

### 5.4 Start Infrastructure Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Check individual services
docker-compose logs postgres
docker-compose logs redis
docker-compose logs qdrant
docker-compose logs minio
```

### 5.5 Verify Services

```bash
# PostgreSQL
psql -h localhost -U postgres -d hr_assistant -c "SELECT version();"
# Password: postgres_password_change_me

# Redis
redis-cli -h localhost -a redis_password_change_me ping
# Expected: PONG

# Qdrant
curl http://localhost:6333/
# Expected: JSON response with version info

# MinIO
curl http://localhost:9000/minio/health/live
# Expected: OK

# Access MinIO Console
open http://localhost:9001
# Login: minio_admin / minio_password_change_me

# Access Grafana
open http://localhost:3001
# Login: admin / grafana_password_change_me
```

---

## 6. Application Configuration

### 6.1 Create Environment Configuration

```bash
cat > .env.example << 'EOF'
# Application
APP_NAME="HR Assistant RAG"
APP_VERSION="1.0.0"
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres_password_change_me@localhost:5432/hr_assistant
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://:redis_password_change_me@localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# Object Storage (MinIO)
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minio_admin
S3_SECRET_KEY=minio_password_change_me
S3_BUCKET_NAME=hr-documents
S3_REGION=us-east-1
S3_USE_SSL=false

# Vector Store
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=hr_policies
FAISS_INDEX_PATH=./data/indices/faiss_index

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=org-your-org-id-optional

# LLM Settings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=500
LLM_TIMEOUT_SECONDS=30

# Embedding Settings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100

# Authentication
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=1
JWT_REFRESH_DAYS=30

# Encryption
ENCRYPTION_KEY=your-32-character-encryption-key

# Rate Limiting
RATE_LIMIT_ENABLED=true
DEFAULT_RATE_LIMIT=60

# Document Processing
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=.pdf,.docx,.txt,.html,.md
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
ENABLE_PII_DETECTION=true

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8000

# Optional: Anthropic (fallback)
ANTHROPIC_API_KEY=

# Optional: Cohere
COHERE_API_KEY=
EOF

# Copy to actual .env file
cp .env.example .env

# Edit .env with your actual values
# IMPORTANT: Replace API keys and passwords!
nano .env
# or
vim .env
```

### 6.2 Create Configuration Module

```bash
cat > config/settings.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "HR Assistant RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = 50

    # Object Storage
    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET_NAME: str = "hr-documents"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False

    # Vector Store
    VECTOR_STORE_TYPE: str = "qdrant"
    QDRANT_URL: Optional[str] = None
    QDRANT_COLLECTION: str = "hr_policies"
    FAISS_INDEX_PATH: Optional[str] = None

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_ORG_ID: Optional[str] = None

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500
    LLM_TIMEOUT_SECONDS: int = 30

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100

    # Authentication
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 1
    JWT_REFRESH_DAYS: int = 30

    # Encryption
    ENCRYPTION_KEY: str

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    DEFAULT_RATE_LIMIT: int = 60

    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: str = ".pdf,.docx,.txt,.html,.md"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Security
    CORS_ORIGINS: str = "*"
    ENABLE_PII_DETECTION: bool = True

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 8000

    # Optional providers
    ANTHROPIC_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None

    @property
    def allowed_file_types_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_FILE_TYPES.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
EOF
```

### 6.3 Verify Configuration

```bash
# Test configuration loading
python -c "
from config.settings import settings
print(f'App: {settings.APP_NAME}')
print(f'Environment: {settings.ENVIRONMENT}')
print(f'Database: {settings.DATABASE_URL[:50]}...')
print(f'OpenAI Key: {settings.OPENAI_API_KEY[:10]}...')
"
```

---

## 7. Database Setup

### 7.1 Create Database Models

```bash
cat > app/models/base.py << 'EOF'
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def get_db():
    """Dependency for getting database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
EOF
```

### 7.2 Create Document Model

```bash
cat > app/models/document.py << 'EOF'
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid as uuid_lib
from .base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    title = Column(String(255), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)
    mime_type = Column(String(100), nullable=False)

    # Metadata
    category = Column(String(100), nullable=False, index=True)
    department = Column(String(100), index=True)
    tags = Column(ARRAY(String), default=[])
    effective_date = Column(DateTime)
    expiry_date = Column(DateTime)
    version = Column(String(20), default="1.0")
    language = Column(String(10), default="en")

    # Processing status
    status = Column(String(20), default="pending", index=True)
    error_message = Column(String(500))
    retry_count = Column(Integer, default=0)

    # Statistics
    chunk_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    query_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)

    # Audit
    uploaded_by = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)

    # Relationships
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title})>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_hash = Column(String(64), nullable=False)

    # Content
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)

    # Position info
    page_number = Column(Integer)
    section_title = Column(String(255))
    start_char = Column(Integer)
    end_char = Column(Integer)

    # Vector reference
    vector_id = Column(String(100), nullable=False)
    embedding_model = Column(String(100), default="text-embedding-3-small")
    embedding_dimension = Column(Integer, default=1536)

    # Metadata
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document_index", "document_id", "chunk_index"),
    )

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, index={self.chunk_index})>"
EOF
```

### 7.3 Setup Alembic Migrations

```bash
# Initialize Alembic
alembic init migrations

# Configure Alembic
cat > migrations/env.py << 'EOF'
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import your models
from app.models.base import Base
from app.models.document import Document, DocumentChunk
from config.settings import settings

# this is the Alembic Config object
config = context.config

# Set the database URL from settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF

# Update alembic.ini
sed -i '' 's|sqlalchemy.url = .*|sqlalchemy.url = |g' alembic.ini
```

### 7.4 Create Initial Migration

```bash
# Generate migration
alembic revision --autogenerate -m "Initial schema"

# Run migration
alembic upgrade head

# Verify tables created
psql -h localhost -U postgres -d hr_assistant -c "\dt"
```

---

## 8. Vector Store Configuration

### 8.1 Initialize Qdrant Collection

```bash
cat > scripts/init_qdrant.py << 'EOF'
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config.settings import settings


async def init_qdrant():
    """Initialize Qdrant collection for HR policies."""
    client = QdrantClient(url=settings.QDRANT_URL)

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if settings.QDRANT_COLLECTION not in collection_names:
        # Create collection
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {settings.QDRANT_COLLECTION}")
    else:
        print(f"Collection already exists: {settings.QDRANT_COLLECTION}")

    # Get collection info
    info = client.get_collection(settings.QDRANT_COLLECTION)
    print(f"Collection info: {info}")


if __name__ == "__main__":
    asyncio.run(init_qdrant())
EOF

# Run initialization
python scripts/init_qdrant.py
```

### 8.2 Initialize MinIO Bucket

```bash
cat > scripts/init_minio.py << 'EOF'
from minio import Minio
from config.settings import settings


def init_minio():
    """Initialize MinIO bucket for document storage."""
    # Parse endpoint
    endpoint = settings.S3_ENDPOINT.replace("http://", "").replace("https://", "")

    client = Minio(
        endpoint,
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
        secure=settings.S3_USE_SSL,
    )

    # Create bucket if not exists
    if not client.bucket_exists(settings.S3_BUCKET_NAME):
        client.make_bucket(settings.S3_BUCKET_NAME)
        print(f"Created bucket: {settings.S3_BUCKET_NAME}")
    else:
        print(f"Bucket already exists: {settings.S3_BUCKET_NAME}")

    # List buckets
    buckets = client.list_buckets()
    print("Available buckets:")
    for bucket in buckets:
        print(f"  - {bucket.name}")


if __name__ == "__main__":
    init_minio()
EOF

# Run initialization
python scripts/init_minio.py
```

---

## 9. Running the Application

### 9.1 Create Main Application

```bash
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from config.settings import settings
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise HR Assistant with RAG capabilities",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
if settings.PROMETHEUS_ENABLED:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("application_starting", environment=settings.ENVIRONMENT)
    # Initialize connections, load models, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("application_shutting_down")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
    }


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Add checks for dependencies here
    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.API_WORKERS,
    )
EOF
```

### 9.2 Run Development Server

```bash
# Option 1: Run directly
python -m app.main

# Option 2: Run with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Run with auto-reload in debug mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
```

### 9.3 Verify Application

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy","version":"1.0.0"}

# Root endpoint
curl http://localhost:8000/
# Expected: {"name":"HR Assistant RAG","version":"1.0.0","environment":"development"}

# Metrics endpoint
curl http://localhost:8000/metrics
# Expected: Prometheus metrics

# OpenAPI documentation
open http://localhost:8000/docs
# Interactive API documentation
```

### 9.4 Run with Docker

```bash
# Build API image
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Build image
docker build -t hr-assistant-api:latest .

# Run container
docker run -p 8000:8000 \
  --env-file .env \
  --network hr-assistant_hr_network \
  hr-assistant-api:latest
```

---

## 10. Monitoring & Observability Setup

### 10.1 Configure Grafana Dashboards

```bash
# Access Grafana
open http://localhost:3001
# Login: admin / grafana_password_change_me

# Add Prometheus data source:
# 1. Go to Configuration > Data Sources
# 2. Click "Add data source"
# 3. Select "Prometheus"
# 4. URL: http://prometheus:9090
# 5. Click "Save & Test"
```

### 10.2 Create Dashboard JSON

```bash
cat > config/grafana-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "HR Assistant RAG Metrics",
    "panels": [
      {
        "title": "Query Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))",
            "legendFormat": "p95 latency"
          }
        ]
      },
      {
        "title": "Query Count",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(rag_queries_total[5m]))",
            "legendFormat": "queries/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(rag_queries_total{status='error'}[5m]) / rate(rag_queries_total[5m]) * 100",
            "legendFormat": "error %"
          }
        ]
      }
    ]
  }
}
EOF
```

### 10.3 Setup Structured Logging

```bash
# Create log directory
mkdir -p data/logs

# Configure log rotation
cat > config/logging.json << 'EOF'
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "json": {
      "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
      "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "DEBUG",
      "formatter": "json",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "INFO",
      "formatter": "json",
      "filename": "data/logs/app.log",
      "maxBytes": 10485760,
      "backupCount": 5
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console", "file"]
  }
}
EOF
```

---

## 11. Production Deployment

### 11.1 Generate Secure Secrets

```bash
# Generate JWT secret
python -c "import secrets; print(secrets.token_urlsafe(64))"

# Generate encryption key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate strong passwords
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 11.2 Create Production Environment File

```bash
cat > .env.production << 'EOF'
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database (use environment variables in deployment)
DATABASE_URL=${DATABASE_URL}
DATABASE_POOL_SIZE=50

# Redis
REDIS_URL=${REDIS_URL}

# Object Storage
S3_ENDPOINT=${S3_ENDPOINT}
S3_ACCESS_KEY=${S3_ACCESS_KEY}
S3_SECRET_KEY=${S3_SECRET_KEY}
S3_USE_SSL=true

# OpenAI
OPENAI_API_KEY=${OPENAI_API_KEY}

# Security (inject from secret manager)
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# Rate Limiting
RATE_LIMIT_ENABLED=true
DEFAULT_RATE_LIMIT=100

# CORS (restrict to your domains)
CORS_ORIGINS=https://hr-assistant.company.com
EOF
```

### 11.3 Kubernetes Deployment

```bash
cat > k8s/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hr-assistant-api
  labels:
    app: hr-assistant-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hr-assistant-api
  template:
    metadata:
      labels:
        app: hr-assistant-api
    spec:
      containers:
        - name: api
          image: your-registry/hr-assistant-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: hr-assistant-secrets
                  key: database-url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: hr-assistant-secrets
                  key: redis-url
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: hr-assistant-secrets
                  key: openai-api-key
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: hr-assistant-secrets
                  key: jwt-secret
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /live
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hr-assistant-api
spec:
  selector:
    app: hr-assistant-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
EOF
```

---

## 12. Troubleshooting

### 12.1 Common Issues and Solutions

#### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
psql -h localhost -U postgres -d hr_assistant

# Reset database (WARNING: destroys data)
docker-compose down -v
docker-compose up -d postgres
alembic upgrade head
```

#### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli -h localhost -a redis_password_change_me ping

# Clear Redis cache
redis-cli -h localhost -a redis_password_change_me FLUSHALL
```

#### Qdrant Issues

```bash
# Check Qdrant is running
curl http://localhost:6333/

# Recreate collection
python scripts/init_qdrant.py
```

#### OpenAI API Issues

```bash
# Test API key
python -c "
from openai import OpenAI
client = OpenAI()
models = client.models.list()
print('API key is valid')
"

# Check rate limits
# Go to https://platform.openai.com/usage
```

#### Docker Resource Issues

```bash
# Check Docker resources
docker system df

# Clean up unused resources
docker system prune -a

# Increase Docker memory (in Docker Desktop)
# Settings > Resources > Memory: 8GB+
```

### 12.2 Useful Commands

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f qdrant

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: destroys data)
docker-compose down -v

# Check Python dependencies
pip list
pip check

# Run tests
pytest tests/ -v

# Format code
black app/
isort app/

# Type checking
mypy app/

# Security scanning
bandit -r app/
safety check
```

### 12.3 Performance Optimization

```bash
# Monitor memory usage
docker stats

# Check database slow queries
psql -h localhost -U postgres -d hr_assistant -c "
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"

# Optimize FAISS index
python -c "
import faiss
index = faiss.read_index('data/indices/faiss_index')
print(f'Index type: {type(index)}')
print(f'Total vectors: {index.ntotal}')
"
```

---

## Next Steps

After completing this installation guide, you should:

1. **Implement API endpoints** - Build the actual RAG pipeline endpoints
2. **Add authentication** - Implement JWT and API key authentication
3. **Create document processing workers** - Build async document ingestion
4. **Setup CI/CD pipeline** - Automate testing and deployment
5. **Configure production monitoring** - Set up alerts and dashboards
6. **Load test the system** - Use tools like Locust or k6
7. **Document API** - Complete OpenAPI specifications
8. **Security audit** - Run penetration testing
9. **Backup strategy** - Implement database and vector store backups
10. **Disaster recovery** - Plan for failure scenarios

---

## Quick Reference Commands

```bash
# Start all services
docker-compose up -d

# Start application
uvicorn app.main:app --reload --port 8000

# Run migrations
alembic upgrade head

# Initialize vector store
python scripts/init_qdrant.py

# Initialize object storage
python scripts/init_minio.py

# Run tests
pytest tests/ -v --cov=app

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

---

## Support

For issues and questions:

- Check the [Troubleshooting](#12-troubleshooting) section
- Review logs: `docker-compose logs`
- Open an issue on GitHub
- Contact the development team

---

**Happy Building!** 🚀
