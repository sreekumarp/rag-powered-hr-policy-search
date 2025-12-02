# HR Assistant RAG - Enterprise Edition v3.0.0

A production-ready RAG (Retrieval-Augmented Generation) system for HR policy management with enterprise features including authentication, role-based access control, rate limiting, and comprehensive analytics.

## Features

### Core Capabilities
- **Semantic Search**: Vector-based document search using FAISS and Sentence Transformers
- **Document Management**: Upload and index PDF, DOCX, TXT, HTML, and Markdown files
- **Query Analytics**: Track confidence scores, latency, and usage patterns
- **Metadata Filtering**: Filter queries by category, department, and custom tags

### Enterprise Features
- **Authentication**: JWT tokens and API key support
- **Authorization**: Role-based access control (Admin, User, Readonly)
- **Rate Limiting**: Token bucket algorithm with per-user/per-endpoint limits
- **Database**: SQLAlchemy ORM with SQLite/PostgreSQL support
- **Logging**: Structured JSON logging with rotation
- **Health Checks**: Built-in health endpoints for monitoring
- **Docker**: Production-ready containerization with Docker Compose

### Architecture Highlights
- **Modular Design**: Separation of concerns (models, services, routes, middleware)
- **RESTful API**: Clean REST API with OpenAPI-compatible design
- **Database Migrations**: Alembic for schema versioning
- **CLI Tools**: Management commands for user administration
- **Testing**: Comprehensive test suite with pytest

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

### TL;DR - Local Development

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install 'huggingface_hub<0.17' click

# Configure
cp .env.example .env

# Initialize
python manage.py init-db
python manage.py create-admin

# Run
export PORT=5001  # Or use 51425
python app.py
```

### TL;DR - Docker

```bash
cp .env.example .env
# Edit .env with secure passwords!
docker-compose up --build
```

## Project Structure

```
backend/
├── app.py                          # Main Flask application
├── config.py                       # Environment-based configuration
├── manage.py                       # CLI management commands
├── models/                         # SQLAlchemy database models
│   ├── __init__.py
│   ├── user.py                     # User model with authentication
│   ├── document.py                 # Document metadata model
│   ├── query_log.py                # Query logging and analytics
│   └── api_key.py                  # API key management
├── services/                       # Business logic layer
│   ├── __init__.py
│   ├── auth_service.py             # Authentication and user management
│   ├── document_service.py         # Document upload and indexing
│   └── rag_service.py              # RAG query processing and logging
├── routes/                         # API route blueprints
│   ├── __init__.py
│   ├── auth_routes.py              # /api/auth/* endpoints
│   ├── document_routes.py          # /api/documents/* endpoints
│   ├── query_routes.py             # /api/query endpoints
│   └── admin_routes.py             # /api/admin/* endpoints
├── middleware/                     # Request middleware
│   ├── __init__.py
│   ├── auth.py                     # JWT and API key authentication
│   └── rate_limiter.py             # Token bucket rate limiting
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── vector_store.py             # FAISS vector store with metadata
│   ├── rag_pipeline.py             # RAG query processing
│   ├── document_processor.py       # Multi-format document parser
│   └── logger.py                   # Structured JSON logging
├── migrations/                     # Alembic database migrations
│   ├── alembic.ini
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── data/                           # Data directory (FAISS index, DB)
│   ├── sample_faq.txt              # Sample HR FAQ content
│   ├── faiss_index.faiss           # FAISS index (generated)
│   ├── faiss_index_docs.pkl        # Document chunks (generated)
│   ├── faiss_index_metadata.json   # Metadata (generated)
│   └── hr_assistant.db             # SQLite database (generated)
├── uploads/                        # Uploaded documents
├── logs/                           # Application logs
│   ├── app.log                     # Main log file (JSON)
│   └── error.log                   # Error log file (JSON)
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   └── test_app.py
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Docker Compose with PostgreSQL + Redis
├── .dockerignore
├── .gitignore
├── QUICKSTART.md                   # Quick start guide
└── README.md                       # This file
```

## API Documentation

### Authentication Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/register` | Register new user | None |
| POST | `/api/auth/login` | Login and get JWT tokens | None |
| POST | `/api/auth/refresh` | Refresh access token | None |
| GET | `/api/auth/me` | Get current user profile | JWT |
| POST | `/api/auth/change-password` | Change password | JWT |
| GET | `/api/auth/api-keys` | List user's API keys | JWT |
| POST | `/api/auth/api-keys` | Create new API key | JWT |
| DELETE | `/api/auth/api-keys/<id>` | Revoke API key | JWT |

### Document Endpoints

| Method | Endpoint | Description | Auth | Permission |
|--------|----------|-------------|------|------------|
| GET | `/api/documents` | List documents | JWT/API Key | read |
| GET | `/api/documents/<source_id>` | Get document details | JWT/API Key | read |
| POST | `/api/documents` | Upload document | JWT/API Key | write |
| DELETE | `/api/documents/<source_id>` | Delete document | JWT/API Key | delete |
| PATCH | `/api/documents/<source_id>` | Update metadata | JWT/API Key | write |
| GET | `/api/documents/statistics` | Document statistics | JWT/API Key | read |
| POST | `/api/documents/index-sample` | Index sample FAQ | JWT/API Key | write |

### Query Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/query` | Query HR policies | Optional |
| GET | `/api/query/history` | User's query history | JWT/API Key |
| GET | `/api/analytics` | Query analytics | JWT/API Key |
| GET | `/api/analytics/popular` | Popular queries | JWT/API Key |
| GET | `/api/analytics/low-confidence` | Low confidence queries | JWT/API Key |

### Admin Endpoints

| Method | Endpoint | Description | Auth | Permission |
|--------|----------|-------------|------|------------|
| GET | `/api/admin/users` | List all users | JWT | admin |
| GET | `/api/admin/users/<id>` | Get user details | JWT | admin |
| PATCH | `/api/admin/users/<id>/role` | Update user role | JWT | admin |
| POST | `/api/admin/users/<id>/deactivate` | Deactivate user | JWT | admin |
| GET | `/api/admin/stats` | System statistics | JWT | admin |
| DELETE | `/api/admin/clear-data` | Clear all data | JWT | admin |
| GET | `/api/admin/logs/queries` | Query logs | JWT | admin |
| GET | `/api/admin/config` | System configuration | JWT | admin |

### Public Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Health check | None |
| GET | `/api/info` | API information | None |

## Configuration

Configuration is managed through environment variables. See [.env.example](.env.example) for all options.

### Key Configuration Options

```bash
# Security
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
ADMIN_PASSWORD=admin123

# Database
DATABASE_URL=sqlite:///data/hr_assistant.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost:5432/hr_assistant

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100/hour
RATE_LIMIT_QUERY=50/hour
RATE_LIMIT_UPLOAD=10/hour

# Vector Store
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=3
CONFIDENCE_THRESHOLD=0.3
MIN_CONFIDENCE=0.5

# File Upload
MAX_UPLOAD_SIZE_MB=50

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## User Roles and Permissions

### Roles

- **admin**: Full system access including user management
- **user**: Can upload documents, query, and manage own API keys
- **readonly**: Can only query the system

### Permissions

- **read**: Query documents and view analytics
- **write**: Upload and index documents
- **delete**: Delete documents
- **admin**: Manage users and system configuration

## CLI Management Commands

```bash
# User Management
python manage.py create-admin
python manage.py create-user <username> <email> <password> --role user
python manage.py list-users
python manage.py change-role <user_id> <new_role>
python manage.py deactivate <user_id>
python manage.py create-api-key <username>

# Database
python manage.py init-db
python manage.py reset-db  # CAUTION: Deletes all data!

# Statistics
python manage.py stats
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_vector_store.py -v
```

### Code Formatting

```bash
# Format code with black
black .

# Lint with flake8
flake8 .
```

### Database Migrations

```bash
# Initialize migrations (first time only)
flask db init

# Create migration after model changes
flask db migrate -m "Description of changes"

# Apply migrations
flask db upgrade

# Rollback migration
flask db downgrade
```

## Deployment

### Docker Compose (Recommended)

1. Configure environment:
```bash
cp .env.example .env
# Edit .env with production values
```

2. Start services:
```bash
docker-compose up -d --build
```

3. Create admin user:
```bash
docker-compose exec api python manage.py create-admin
```

4. Monitor logs:
```bash
docker-compose logs -f api
```

### Manual Deployment

1. Use PostgreSQL for the database
2. Use Redis for distributed rate limiting
3. Use a reverse proxy (nginx) with SSL
4. Set up monitoring and log aggregation
5. Configure backup strategies for data and uploads
6. Use environment-specific configuration

## Performance Considerations

- **First Load**: 10-30 seconds (downloads embedding model)
- **Document Indexing**: ~1-2 seconds per page
- **Query Latency**: ~100-300ms
- **Memory Usage**: ~1-2GB (embedding model + FAISS index)

## Limitations

- In-memory FAISS index (not distributed)
- Simple response generation (no LLM integration)
- File deletion requires manual index cleanup
- No streaming responses
- SQLite not recommended for high concurrency

## Roadmap

- [ ] LLM integration (OpenAI/Claude) for better responses
- [ ] Streaming query responses
- [ ] Vector database (Qdrant/Pinecone) for scalability
- [ ] Async processing with Celery
- [ ] Redis caching for frequently asked questions
- [ ] React/Next.js frontend
- [ ] Kubernetes deployment manifests
- [ ] Multi-tenancy support
- [ ] Advanced analytics dashboard

## Troubleshooting

### Common Issues

**Port 51425 in use on macOS**
```bash
export PORT=5001
python app.py
```

**huggingface_hub ImportError**
```bash
pip install 'huggingface_hub<0.17'
```

**Database locked (SQLite)**
- Use PostgreSQL for production
- Ensure only one process accesses SQLite

**Rate limit errors in development**
```bash
# Disable in .env
RATE_LIMIT_ENABLED=false
```

**FAISS index not persisting**
- Check write permissions on `data/` directory
- Verify FAISS_INDEX_PATH in configuration

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check logs in `logs/app.log`
- Run `python manage.py stats` for system health
- Review API documentation above
- See QUICKSTART.md for setup help
