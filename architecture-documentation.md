# Enterprise HR Assistant with RAG - Architecture & Design Documentation

---

## Document 1: System Architecture

### 1.1 High-Level System Design

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Application]
        MOBILE[Mobile App]
        API_CLIENT[API Clients]
    end

    subgraph "API Gateway"
        GATEWAY[Kong/Nginx Gateway]
        AUTH[Auth Service]
        RATE[Rate Limiter]
    end

    subgraph "Application Layer"
        API[Flask/FastAPI Backend]
        QUEUE[Message Queue - Redis/RabbitMQ]
        WORKER[Background Workers]
    end

    subgraph "Processing Layer"
        DOC_PROC[Document Processor]
        CHUNKER[Text Chunker]
        EMBEDDER[Embedding Service]
    end

    subgraph "Data Layer"
        VECTOR_DB[FAISS/Qdrant Vector Store]
        POSTGRES[PostgreSQL - Metadata]
        REDIS_CACHE[Redis Cache]
        S3[Object Storage - S3/MinIO]
    end

    subgraph "AI Layer"
        LLM[LLM Service - OpenAI/Claude]
        RERANK[Re-ranking Service]
    end

    subgraph "Observability"
        PROM[Prometheus]
        GRAF[Grafana]
        ELK[ELK Stack]
    end

    WEB --> GATEWAY
    MOBILE --> GATEWAY
    API_CLIENT --> GATEWAY

    GATEWAY --> AUTH
    GATEWAY --> RATE
    AUTH --> API
    RATE --> API

    API --> QUEUE
    API --> REDIS_CACHE
    API --> POSTGRES
    API --> VECTOR_DB
    API --> LLM

    QUEUE --> WORKER
    WORKER --> DOC_PROC
    DOC_PROC --> CHUNKER
    CHUNKER --> EMBEDDER
    EMBEDDER --> VECTOR_DB
    EMBEDDER --> POSTGRES

    WORKER --> S3

    API --> PROM
    WORKER --> PROM
    PROM --> GRAF
    API --> ELK
```

### 1.2 Component Breakdown

#### Frontend Layer
- **Web Application**: React/Next.js SPA for HR administrators and employees
- **Mobile App**: React Native for on-the-go access
- **API Clients**: SDK for third-party integrations (Slack, Teams, HRIS)

#### API Gateway
- **Kong/Nginx**: Load balancing, SSL termination, request routing
- **Auth Service**: JWT validation, API key management
- **Rate Limiter**: Token bucket algorithm, per-user/per-endpoint limits

#### Application Layer (Backend API)
```python
# Core API structure
hr_assistant/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── documents.py      # Document CRUD operations
│   │   ├── queries.py        # RAG query endpoints
│   │   ├── admin.py          # Administrative operations
│   │   └── health.py         # Health checks
│   ├── middleware/
│   │   ├── auth.py           # Authentication middleware
│   │   ├── rate_limit.py     # Rate limiting
│   │   └── logging.py        # Request/response logging
│   └── schemas/
│       ├── document.py       # Pydantic models
│       ├── query.py
│       └── response.py
├── services/
│   ├── document_service.py   # Document processing logic
│   ├── rag_service.py        # RAG pipeline orchestration
│   ├── embedding_service.py  # Vector embedding generation
│   └── llm_service.py        # LLM interaction
├── workers/
│   ├── document_worker.py    # Async document processing
│   └── reindex_worker.py     # Batch reindexing
├── models/
│   ├── document.py           # SQLAlchemy models
│   ├── user.py
│   └── audit.py
└── config/
    ├── settings.py           # Configuration management
    └── logging.py
```

#### Document Processing Layer
- **Document Processor**: Multi-format parser (PDF, DOCX, TXT, HTML)
- **Text Chunker**: Semantic chunking with overlap and metadata preservation
- **Embedding Service**: Batch embedding generation with caching

#### Vector Store
- **Primary**: FAISS for in-memory vector search (< 10M vectors)
- **Alternative**: Qdrant/Milvus for distributed deployments
- **Persistence**: Periodic snapshots to object storage

#### LLM Integration
- **Primary**: OpenAI GPT-4o / GPT-4o-mini
- **Fallback**: Anthropic Claude 3.5 Sonnet
- **On-premise option**: Llama 3 70B via vLLM

#### Database
- **PostgreSQL**: Metadata, user data, audit logs, document tracking
- **Redis**: Caching, session management, rate limiting state
- **Object Storage**: Raw document storage, vector index backups

### 1.3 Data Flow Diagrams

#### Document Upload → Indexing Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Queue
    participant Worker
    participant S3
    participant Processor
    participant Embedder
    participant VectorDB
    participant Postgres

    Client->>API: POST /upload (file)
    API->>API: Validate file type & size
    API->>S3: Store raw document
    API->>Postgres: Create document record (status: pending)
    API->>Queue: Enqueue processing job
    API-->>Client: 202 Accepted (job_id)

    Queue->>Worker: Dequeue job
    Worker->>S3: Fetch document
    Worker->>Processor: Parse document
    Processor->>Processor: Extract text & metadata
    Processor->>Worker: Return parsed content

    Worker->>Processor: Chunk text
    Processor->>Processor: Apply chunking strategy
    Processor-->>Worker: Return chunks[]

    loop For each chunk batch
        Worker->>Embedder: Generate embeddings
        Embedder->>Embedder: Call embedding API
        Embedder-->>Worker: Return vectors[]
    end

    Worker->>VectorDB: Index vectors with metadata
    Worker->>Postgres: Store chunk metadata
    Worker->>Postgres: Update document status (indexed)
    Worker->>Queue: Publish completion event
```

#### Query → Retrieval → Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Cache
    participant Embedder
    participant VectorDB
    participant Reranker
    participant LLM
    participant Postgres

    Client->>API: POST /query {question, filters}
    API->>API: Validate & sanitize input
    API->>Cache: Check query cache

    alt Cache Hit
        Cache-->>API: Return cached response
        API-->>Client: 200 OK (cached: true)
    else Cache Miss
        API->>Embedder: Embed question
        Embedder-->>API: Query vector

        API->>VectorDB: Search similar vectors (top-k=10)
        VectorDB-->>API: Retrieved chunks[]

        API->>Reranker: Re-rank by relevance
        Reranker-->>API: Top-4 chunks

        API->>API: Build prompt with context
        API->>LLM: Generate response
        LLM-->>API: Answer text

        API->>API: Validate & format response
        API->>Cache: Store in cache (TTL: 1h)
        API->>Postgres: Log query & response
        API-->>Client: 200 OK {answer, sources, confidence}
    end
```

### 1.4 API Contract Specifications (OpenAPI 3.0)

```yaml
openapi: 3.0.3
info:
  title: HR Assistant RAG API
  version: 1.0.0
  description: Enterprise HR Policy Assistant with RAG capabilities

servers:
  - url: https://api.hr-assistant.company.com/v1
    description: Production
  - url: https://api-staging.hr-assistant.company.com/v1
    description: Staging

security:
  - BearerAuth: []
  - ApiKeyAuth: []

paths:
  /upload:
    post:
      summary: Upload HR document for indexing
      operationId: uploadDocument
      tags: [Documents]
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                metadata:
                  type: object
                  properties:
                    title: {type: string}
                    category: {type: string}
                    effective_date: {type: string, format: date}
                    department: {type: string}
      responses:
        '202':
          description: Document accepted for processing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentUploadResponse'
        '400': {$ref: '#/components/responses/BadRequest'}
        '413': {$ref: '#/components/responses/PayloadTooLarge'}
        '429': {$ref: '#/components/responses/RateLimited'}

  /query:
    post:
      summary: Query HR policies using natural language
      operationId: queryPolicies
      tags: [Queries]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
      responses:
        '200':
          description: Successful query response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResponse'
        '400': {$ref: '#/components/responses/BadRequest'}
        '429': {$ref: '#/components/responses/RateLimited'}

  /documents:
    get:
      summary: List all indexed documents
      operationId: listDocuments
      tags: [Documents]
      parameters:
        - name: page
          in: query
          schema: {type: integer, default: 1}
        - name: limit
          in: query
          schema: {type: integer, default: 20, maximum: 100}
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, processing, indexed, failed]
        - name: category
          in: query
          schema: {type: string}
      responses:
        '200':
          description: List of documents
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentListResponse'

  /documents/{id}:
    delete:
      summary: Remove document from index
      operationId: deleteDocument
      tags: [Documents]
      parameters:
        - name: id
          in: path
          required: true
          schema: {type: string, format: uuid}
      responses:
        '204': {description: Document deleted}
        '404': {$ref: '#/components/responses/NotFound'}

  /reindex:
    post:
      summary: Trigger batch reindexing
      operationId: reindexDocuments
      tags: [Admin]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                document_ids:
                  type: array
                  items: {type: string, format: uuid}
                force:
                  type: boolean
                  default: false
      responses:
        '202':
          description: Reindexing job started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReindexResponse'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    DocumentUploadResponse:
      type: object
      properties:
        document_id: {type: string, format: uuid}
        job_id: {type: string, format: uuid}
        status: {type: string, enum: [pending]}
        message: {type: string}

    QueryRequest:
      type: object
      required: [question]
      properties:
        question:
          type: string
          minLength: 3
          maxLength: 1000
        filters:
          type: object
          properties:
            categories: {type: array, items: {type: string}}
            date_range:
              type: object
              properties:
                start: {type: string, format: date}
                end: {type: string, format: date}
        max_sources: {type: integer, default: 4, maximum: 10}
        include_confidence: {type: boolean, default: true}

    QueryResponse:
      type: object
      properties:
        answer: {type: string}
        sources:
          type: array
          items:
            $ref: '#/components/schemas/Source'
        confidence: {type: number, minimum: 0, maximum: 1}
        processing_time_ms: {type: integer}
        cached: {type: boolean}
        query_id: {type: string, format: uuid}

    Source:
      type: object
      properties:
        document_id: {type: string, format: uuid}
        document_title: {type: string}
        chunk_text: {type: string}
        relevance_score: {type: number}
        page_number: {type: integer}

    DocumentListResponse:
      type: object
      properties:
        documents:
          type: array
          items:
            $ref: '#/components/schemas/Document'
        pagination:
          $ref: '#/components/schemas/Pagination'

    Document:
      type: object
      properties:
        id: {type: string, format: uuid}
        title: {type: string}
        filename: {type: string}
        category: {type: string}
        status: {type: string}
        chunk_count: {type: integer}
        uploaded_at: {type: string, format: date-time}
        indexed_at: {type: string, format: date-time}

    Pagination:
      type: object
      properties:
        page: {type: integer}
        limit: {type: integer}
        total: {type: integer}
        has_next: {type: boolean}

    ReindexResponse:
      type: object
      properties:
        job_id: {type: string, format: uuid}
        documents_count: {type: integer}
        estimated_time_seconds: {type: integer}

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            type: object
            properties:
              error: {type: string}
              details: {type: object}
    NotFound:
      description: Resource not found
    PayloadTooLarge:
      description: File size exceeds limit
    RateLimited:
      description: Rate limit exceeded
      headers:
        X-RateLimit-Limit: {schema: {type: integer}}
        X-RateLimit-Remaining: {schema: {type: integer}}
        X-RateLimit-Reset: {schema: {type: integer}}
```

### 1.5 Technology Stack Rationale

| Component | Technology | Rationale | Alternatives Considered |
|-----------|-----------|-----------|------------------------|
| **Backend Framework** | FastAPI | Async support, auto-docs, Pydantic validation, high performance | Flask (simpler but slower), Django (overkill for API) |
| **Vector Store** | FAISS + Qdrant | FAISS for speed, Qdrant for persistence & filtering | Pinecone (costly), Milvus (complex setup) |
| **Database** | PostgreSQL 15 | ACID compliance, JSON support, mature ecosystem | MySQL (less features), MongoDB (not needed) |
| **Cache** | Redis 7 | In-memory speed, pub/sub, rate limiting | Memcached (less features) |
| **Message Queue** | Redis Streams | Simplicity, already have Redis | RabbitMQ (more complex), Kafka (overkill) |
| **LLM Provider** | OpenAI GPT-4o-mini | Cost-effective, high quality, fast | GPT-4o (costlier), Claude (context but cost) |
| **Embeddings** | OpenAI text-embedding-3-small | Quality/cost balance, 1536 dims | BGE-large (slower), Cohere (alternative) |
| **Object Storage** | MinIO (S3-compatible) | Self-hosted, S3 API compatibility | AWS S3 (vendor lock-in) |
| **Container Runtime** | Docker + Kubernetes | Industry standard, auto-scaling | Docker Swarm (less features) |
| **Monitoring** | Prometheus + Grafana | Open-source, extensive ecosystem | DataDog (costly), CloudWatch (AWS-specific) |
| **Logging** | ELK Stack | Powerful search, visualization | Loki (simpler), CloudWatch (AWS-specific) |

### 1.6 Scalability & Performance Targets

#### Performance SLAs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Latency (p50) | < 2 seconds | End-to-end response time |
| Query Latency (p95) | < 5 seconds | Including LLM generation |
| Query Latency (p99) | < 10 seconds | Worst case scenarios |
| Document Indexing | < 30 seconds/page | From upload to searchable |
| Uptime | 99.9% | Monthly availability |
| Error Rate | < 0.1% | Failed requests / total |

#### Scalability Targets

| Scale Level | Documents | Concurrent Users | Infrastructure |
|-------------|-----------|-----------------|----------------|
| **Small** | < 10K | 50 | Single server, 16GB RAM |
| **Medium** | 10K - 100K | 500 | 3-node cluster, 32GB each |
| **Large** | 100K - 1M | 5,000 | K8s cluster, auto-scaling |
| **Enterprise** | > 1M | 50,000+ | Multi-region, sharded |

#### Scaling Strategy

1. **Horizontal Scaling**
   - Stateless API servers behind load balancer
   - Multiple worker instances for document processing
   - Sharded vector indices across nodes

2. **Vertical Scaling**
   - Increase RAM for larger in-memory indices
   - GPU acceleration for embedding generation
   - SSD storage for vector index persistence

3. **Caching Strategy**
   - Query result caching (1-hour TTL)
   - Embedding caching (permanent)
   - Document metadata caching

4. **Cost vs. Performance Trade-offs**

```
Cost Tier: Budget ($100-300/month)
├── Single server deployment
├── GPT-4o-mini exclusively
├── In-memory FAISS (no persistence)
├── Basic monitoring (logs only)
└── Max: 10K documents, 50 users

Cost Tier: Standard ($500-1500/month)
├── 3-node cluster with Redis HA
├── GPT-4o-mini + GPT-4o routing
├── Qdrant with persistence
├── Full observability stack
└── Max: 100K documents, 500 users

Cost Tier: Enterprise ($3000+/month)
├── Kubernetes with auto-scaling
├── Multi-model routing + fallbacks
├── Distributed vector store (Milvus)
├── Multi-region deployment
├── 24/7 monitoring & alerting
└── Max: 1M+ documents, 5000+ users
```

---

## Document 2: API Specification

### 2.1 Endpoint Details

#### POST /upload - Document Ingestion

**Purpose**: Upload HR policy documents for processing and indexing

**Request**:
```http
POST /v1/upload HTTP/1.1
Host: api.hr-assistant.company.com
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="leave_policy_2024.pdf"
Content-Type: application/pdf

<binary_data>
------WebKitFormBoundary
Content-Disposition: form-data; name="metadata"
Content-Type: application/json

{
  "title": "Annual Leave Policy 2024",
  "category": "leave_policies",
  "effective_date": "2024-01-01",
  "department": "all",
  "tags": ["leave", "vacation", "pto"],
  "version": "2.0"
}
------WebKitFormBoundary--
```

**Response** (202 Accepted):
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "pending",
  "message": "Document accepted for processing",
  "estimated_processing_time_seconds": 45,
  "webhook_url": "/v1/jobs/7c9e6679-7425-40de-944b-e07fc1f90ae7/status"
}
```

**Validation Rules**:
- Max file size: 50MB
- Allowed types: PDF, DOCX, DOC, TXT, HTML, MD
- Required metadata: title, category
- Title length: 3-255 characters

---

#### POST /query - FAQ Queries

**Purpose**: Query HR policies using natural language

**Request**:
```json
{
  "question": "How many days of annual leave am I entitled to?",
  "filters": {
    "categories": ["leave_policies"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "departments": ["engineering"]
  },
  "max_sources": 4,
  "include_confidence": true,
  "conversation_id": "session_123"
}
```

**Response** (200 OK):
```json
{
  "answer": "According to the Annual Leave Policy 2024, full-time employees are entitled to:\n\n- **Standard entitlement**: 20 days of paid annual leave per calendar year\n- **Pro-rated for new joiners**: Based on joining date\n- **Carryover**: Up to 5 days can be carried forward to the next year\n\nNote: Engineering department employees receive an additional 2 days for on-call compensation, totaling 22 days.",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_title": "Annual Leave Policy 2024",
      "chunk_text": "Full-time employees are entitled to 20 days of paid annual leave per calendar year...",
      "relevance_score": 0.94,
      "page_number": 3,
      "section": "2.1 Leave Entitlement"
    },
    {
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "document_title": "Engineering Department Handbook",
      "chunk_text": "Engineers participating in on-call rotations receive 2 additional leave days...",
      "relevance_score": 0.87,
      "page_number": 15,
      "section": "5. Compensation & Benefits"
    }
  ],
  "confidence": 0.91,
  "processing_time_ms": 1847,
  "cached": false,
  "query_id": "q_abc123def456",
  "tokens_used": {
    "prompt": 1250,
    "completion": 180,
    "total": 1430
  }
}
```

---

#### GET /documents - List Indexed Documents

**Request**:
```http
GET /v1/documents?page=1&limit=20&status=indexed&category=leave_policies HTTP/1.1
Authorization: Bearer <jwt_token>
```

**Response** (200 OK):
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Annual Leave Policy 2024",
      "filename": "leave_policy_2024.pdf",
      "category": "leave_policies",
      "status": "indexed",
      "file_size_bytes": 245678,
      "chunk_count": 12,
      "uploaded_by": "admin@company.com",
      "uploaded_at": "2024-01-15T10:30:00Z",
      "indexed_at": "2024-01-15T10:30:45Z",
      "last_accessed": "2024-03-20T14:22:00Z",
      "query_count": 156
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  },
  "summary": {
    "total_documents": 45,
    "total_chunks": 892,
    "index_size_mb": 128.5
  }
}
```

---

#### DELETE /documents/{id} - Document Removal

**Request**:
```http
DELETE /v1/documents/550e8400-e29b-41d4-a716-446655440000 HTTP/1.1
Authorization: Bearer <jwt_token>
```

**Response** (204 No Content):
```
(empty body)
```

**Side Effects**:
- Removes document from vector index
- Deletes file from object storage
- Clears related cache entries
- Logs deletion in audit trail

---

#### POST /reindex - Batch Reindexing

**Request**:
```json
{
  "document_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660e8400-e29b-41d4-a716-446655440001"
  ],
  "force": true,
  "new_chunking_strategy": {
    "chunk_size": 1200,
    "chunk_overlap": 250
  }
}
```

**Response** (202 Accepted):
```json
{
  "job_id": "reindex_789xyz",
  "documents_count": 2,
  "total_chunks_to_process": 24,
  "estimated_time_seconds": 120,
  "status_url": "/v1/jobs/reindex_789xyz/status"
}
```

---

### 2.2 Error Codes and Handling Strategy

#### Error Response Schema

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "question",
      "issue": "Question must be between 3 and 1000 characters"
    },
    "request_id": "req_abc123",
    "timestamp": "2024-03-20T14:30:00Z",
    "documentation_url": "https://docs.hr-assistant.com/errors/VALIDATION_ERROR"
  }
}
```

#### Error Code Catalog

| HTTP Status | Error Code | Description | Resolution |
|------------|------------|-------------|------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters | Check request format |
| 400 | `UNSUPPORTED_FILE_TYPE` | File type not allowed | Use PDF, DOCX, TXT |
| 401 | `AUTHENTICATION_FAILED` | Invalid or missing token | Provide valid JWT |
| 403 | `INSUFFICIENT_PERMISSIONS` | User lacks required role | Contact admin |
| 404 | `DOCUMENT_NOT_FOUND` | Document ID doesn't exist | Verify document ID |
| 409 | `DOCUMENT_ALREADY_EXISTS` | Duplicate document | Update existing or use new ID |
| 413 | `FILE_TOO_LARGE` | File exceeds 50MB limit | Compress or split file |
| 422 | `PROCESSING_FAILED` | Document couldn't be parsed | Check file integrity |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| 500 | `INTERNAL_ERROR` | Unexpected server error | Retry or contact support |
| 502 | `LLM_SERVICE_UNAVAILABLE` | LLM provider down | Automatic retry |
| 503 | `SERVICE_OVERLOADED` | High load, try later | Implement backoff |

#### Error Handling Implementation

```python
from enum import Enum
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: dict = {}
    request_id: str
    timestamp: str
    documentation_url: str

class APIError(HTTPException):
    def __init__(self, status_code: int, error_code: ErrorCode, message: str, details: dict = None):
        self.error_code = error_code
        self.error_message = message
        self.error_details = details or {}
        super().__init__(status_code=status_code, detail=self._build_detail())

    def _build_detail(self):
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.error_message,
                "details": self.error_details,
                "documentation_url": f"https://docs.hr-assistant.com/errors/{self.error_code.value}"
            }
        }
```

---

### 2.3 Rate Limiting and Authentication

#### Rate Limiting Configuration

```python
RATE_LIMITS = {
    "anonymous": {
        "requests_per_minute": 10,
        "requests_per_hour": 50,
        "requests_per_day": 100
    },
    "authenticated": {
        "requests_per_minute": 60,
        "requests_per_hour": 500,
        "requests_per_day": 5000
    },
    "premium": {
        "requests_per_minute": 300,
        "requests_per_hour": 3000,
        "requests_per_day": 30000
    },
    "admin": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000,
        "requests_per_day": 100000
    }
}

ENDPOINT_SPECIFIC_LIMITS = {
    "/upload": {
        "requests_per_minute": 10,
        "max_concurrent": 3
    },
    "/query": {
        "requests_per_minute": 30,
        "tokens_per_hour": 100000
    },
    "/reindex": {
        "requests_per_hour": 5
    }
}
```

#### Authentication Methods

**1. JWT Token (Primary)**
```python
# Token payload structure
{
    "sub": "user_id_123",
    "email": "user@company.com",
    "roles": ["hr_admin", "query_user"],
    "org_id": "org_456",
    "iat": 1710936000,
    "exp": 1710939600,
    "jti": "unique_token_id"
}

# Validation
def validate_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
            options={"require_exp": True}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise APIError(401, ErrorCode.AUTHENTICATION_FAILED, "Token expired")
    except jwt.InvalidTokenError:
        raise APIError(401, ErrorCode.AUTHENTICATION_FAILED, "Invalid token")
```

**2. API Key (Service-to-Service)**
```python
# API key format: hrass_<environment>_<random_32_chars>
# Example: hrass_prod_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

def validate_api_key(api_key: str) -> dict:
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_record = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True,
        APIKey.expires_at > datetime.utcnow()
    ).first()

    if not key_record:
        raise APIError(401, ErrorCode.AUTHENTICATION_FAILED, "Invalid API key")

    return {
        "client_id": key_record.client_id,
        "roles": key_record.roles,
        "rate_limit_tier": key_record.tier
    }
```

---

## Document 3: Data Models & Schema

### 3.1 Document Schema

```python
# SQLAlchemy Model
from sqlalchemy import Column, String, Integer, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid
from datetime import datetime

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)  # S3 path
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256
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
    status = Column(
        SQLEnum("pending", "processing", "indexed", "failed", name="doc_status"),
        default="pending",
        index=True
    )
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
    deleted_at = Column(DateTime)  # Soft delete

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
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
    vector_id = Column(String(100), nullable=False)  # ID in FAISS/Qdrant
    embedding_model = Column(String(100), default="text-embedding-3-small")
    embedding_dimension = Column(Integer, default=1536)

    # Metadata
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document_index", "document_id", "chunk_index"),
    )
```

### 3.2 Query Schema

```python
# Pydantic models for API
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date
import uuid

class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None

class QueryFilters(BaseModel):
    categories: Optional[List[str]] = Field(default=[], max_items=10)
    departments: Optional[List[str]] = Field(default=[], max_items=10)
    tags: Optional[List[str]] = Field(default=[], max_items=20)
    date_range: Optional[DateRange] = None
    document_ids: Optional[List[uuid.UUID]] = Field(default=[], max_items=50)
    exclude_expired: bool = True

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    filters: Optional[QueryFilters] = QueryFilters()
    max_sources: int = Field(default=4, ge=1, le=10)
    include_confidence: bool = True
    include_token_usage: bool = False
    conversation_id: Optional[str] = None
    user_context: Optional[dict] = None

    @validator("question")
    def sanitize_question(cls, v):
        # Remove potential injection attempts
        v = v.strip()
        v = re.sub(r"[<>{}]", "", v)
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the sick leave policy?",
                "filters": {
                    "categories": ["leave_policies"],
                    "exclude_expired": True
                },
                "max_sources": 4
            }
        }


# Internal query representation
class InternalQuery(BaseModel):
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    original_question: str
    processed_question: str
    embedding: Optional[List[float]] = None
    filters: QueryFilters
    user_id: str
    org_id: str
    timestamp: datetime
    metadata: dict = {}
```

### 3.3 Response Schema

```python
class Source(BaseModel):
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    chunk_text: str = Field(..., max_length=500)
    relevance_score: float = Field(..., ge=0, le=1)
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = {}

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

class QueryResponse(BaseModel):
    query_id: uuid.UUID
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0, le=1)
    processing_time_ms: int
    cached: bool
    token_usage: Optional[TokenUsage] = None
    warnings: List[str] = []
    metadata: dict = {}

    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "550e8400-e29b-41d4-a716-446655440000",
                "answer": "Employees are entitled to 20 days of annual leave...",
                "sources": [
                    {
                        "document_id": "660e8400-e29b-41d4-a716-446655440001",
                        "document_title": "Leave Policy 2024",
                        "chunk_text": "Full-time employees receive 20 days...",
                        "relevance_score": 0.94,
                        "page_number": 3
                    }
                ],
                "confidence": 0.91,
                "processing_time_ms": 1847,
                "cached": False
            }
        }


# Response for streaming (if implemented)
class StreamingResponseChunk(BaseModel):
    chunk_type: str  # "token", "source", "metadata", "done"
    content: str
    sequence: int
    timestamp: float
```

### 3.4 Storage Schema

```sql
-- PostgreSQL Schema

-- Main documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    department VARCHAR(100),
    tags TEXT[],
    effective_date TIMESTAMP,
    expiry_date TIMESTAMP,
    version VARCHAR(20) DEFAULT '1.0',
    language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(20) DEFAULT 'pending',
    error_message VARCHAR(500),
    retry_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    query_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    uploaded_by VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    indexed_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP
);

CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX idx_documents_effective_date ON documents(effective_date);

-- Document chunks table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_hash VARCHAR(64) NOT NULL,
    text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(255),
    start_char INTEGER,
    end_char INTEGER,
    vector_id VARCHAR(100) NOT NULL,
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    embedding_dimension INTEGER DEFAULT 1536,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_vector_id ON document_chunks(vector_id);
CREATE UNIQUE INDEX idx_chunks_doc_index ON document_chunks(document_id, chunk_index);

-- Query logs for analytics and auditing
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    org_id VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    question_hash VARCHAR(64) NOT NULL,
    filters JSONB DEFAULT '{}',
    answer TEXT,
    sources JSONB DEFAULT '[]',
    confidence FLOAT,
    cached BOOLEAN DEFAULT FALSE,
    processing_time_ms INTEGER,
    tokens_used JSONB,
    error_code VARCHAR(50),
    error_message TEXT,
    client_ip INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_query_logs_user ON query_logs(user_id);
CREATE INDEX idx_query_logs_org ON query_logs(org_id);
CREATE INDEX idx_query_logs_created ON query_logs(created_at);
CREATE INDEX idx_query_logs_hash ON query_logs(question_hash);

-- Audit logs for compliance
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    user_id VARCHAR(255) NOT NULL,
    org_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    client_ip INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_event ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);

-- API keys management
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id VARCHAR(255) NOT NULL,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    roles TEXT[] NOT NULL,
    tier VARCHAR(50) DEFAULT 'authenticated',
    rate_limits JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_client ON api_keys(client_id);

-- Users table
CREATE TABLE users (
    id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    org_id VARCHAR(255) NOT NULL,
    roles TEXT[] NOT NULL,
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

CREATE INDEX idx_users_org ON users(org_id);
CREATE INDEX idx_users_email ON users(email);
```

---

## Document 4: Security & Compliance

### 4.1 Authentication

#### JWT Token Authentication

```python
import jwt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry_hours = 1
        self.refresh_token_days = 30

    def create_access_token(self, user_data: dict) -> str:
        """Generate JWT access token."""
        payload = {
            "sub": user_data["user_id"],
            "email": user_data["email"],
            "roles": user_data["roles"],
            "org_id": user_data["org_id"],
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "jti": str(uuid.uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Generate refresh token for token renewal."""
        payload = {
            "sub": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_days),
            "jti": str(uuid.uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """Validate and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "require_exp": True,
                    "require_iat": True,
                    "verify_exp": True
                }
            )

            # Check if token is blacklisted
            if self._is_token_blacklisted(payload["jti"]):
                raise AuthenticationError("Token has been revoked")

            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    def _is_token_blacklisted(self, jti: str) -> bool:
        """Check Redis for blacklisted tokens."""
        return redis_client.exists(f"blacklist:{jti}")

    def revoke_token(self, jti: str, exp: int):
        """Blacklist a token until its expiry."""
        ttl = exp - int(datetime.utcnow().timestamp())
        if ttl > 0:
            redis_client.setex(f"blacklist:{jti}", ttl, "revoked")
```

#### API Key Management

```python
import secrets
import hashlib

class APIKeyService:
    def generate_api_key(self, client_id: str, name: str, roles: list, tier: str = "authenticated") -> dict:
        """Generate a new API key."""
        # Generate secure random key
        key = f"hrass_prod_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Store hashed key in database
        api_key_record = APIKey(
            client_id=client_id,
            key_hash=key_hash,
            name=name,
            roles=roles,
            tier=tier,
            expires_at=datetime.utcnow() + timedelta(days=365),
            created_by="system"
        )
        db.add(api_key_record)
        db.commit()

        return {
            "api_key": key,  # Only shown once!
            "key_id": str(api_key_record.id),
            "expires_at": api_key_record.expires_at.isoformat()
        }

    def validate_api_key(self, key: str) -> dict:
        """Validate API key and return permissions."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        api_key = db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
            APIKey.expires_at > datetime.utcnow()
        ).first()

        if not api_key:
            raise AuthenticationError("Invalid or expired API key")

        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()
        db.commit()

        return {
            "client_id": api_key.client_id,
            "roles": api_key.roles,
            "tier": api_key.tier,
            "rate_limits": api_key.rate_limits or {}
        }

    def rotate_api_key(self, old_key: str) -> dict:
        """Rotate API key while maintaining grace period."""
        old_hash = hashlib.sha256(old_key.encode()).hexdigest()
        old_record = db.query(APIKey).filter(APIKey.key_hash == old_hash).first()

        if not old_record:
            raise AuthenticationError("API key not found")

        # Create new key
        new_key_data = self.generate_api_key(
            old_record.client_id,
            old_record.name,
            old_record.roles,
            old_record.tier
        )

        # Keep old key active for grace period (24 hours)
        old_record.expires_at = datetime.utcnow() + timedelta(hours=24)
        db.commit()

        return new_key_data
```

### 4.2 Authorization (RBAC)

```python
from functools import wraps
from enum import Enum

class Permission(str, Enum):
    QUERY_READ = "query:read"
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    AUDIT_READ = "audit:read"

ROLE_PERMISSIONS = {
    "query_user": [
        Permission.QUERY_READ,
        Permission.DOCUMENT_READ
    ],
    "hr_admin": [
        Permission.QUERY_READ,
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE
    ],
    "system_admin": [
        Permission.QUERY_READ,
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
        Permission.ADMIN_READ,
        Permission.ADMIN_WRITE,
        Permission.AUDIT_READ
    ]
}

def require_permission(permission: Permission):
    """Decorator to enforce permission checks."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from request context
            user = get_current_user()

            if not has_permission(user.roles, permission):
                raise AuthorizationError(
                    f"Permission denied: {permission.value} required"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def has_permission(user_roles: list, required_permission: Permission) -> bool:
    """Check if user has required permission."""
    for role in user_roles:
        if role in ROLE_PERMISSIONS:
            if required_permission in ROLE_PERMISSIONS[role]:
                return True
    return False

# Usage in endpoints
@app.post("/v1/documents")
@require_permission(Permission.DOCUMENT_WRITE)
async def upload_document(file: UploadFile):
    # Only users with DOCUMENT_WRITE permission can access
    pass
```

### 4.3 Data Privacy & Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class EncryptionService:
    def __init__(self, master_key: str):
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"hr_assistant_salt",  # In production, use secure random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# PII Detection and Redaction
import re

class PIIProtector:
    PII_PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "employee_id": r"\bEMP\d{6}\b"
    }

    def detect_pii(self, text: str) -> dict:
        """Detect PII in text."""
        detected = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
        return detected

    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", redacted)
        return redacted

    def mask_pii_for_logging(self, text: str) -> str:
        """Partially mask PII for logging purposes."""
        masked = text
        # Show first/last chars only
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 4:
                    masked_value = match[0] + "*" * (len(match) - 2) + match[-1]
                else:
                    masked_value = "*" * len(match)
                masked = masked.replace(match, masked_value)
        return masked
```

### 4.4 Audit Logging

```python
from datetime import datetime
import json

class AuditLogger:
    def __init__(self):
        self.pii_protector = PIIProtector()

    def log_event(
        self,
        event_type: str,
        resource_type: str,
        resource_id: str,
        action: str,
        user_id: str,
        org_id: str,
        details: dict = None,
        request_context: dict = None
    ):
        """Log audit event to database."""
        # Sanitize details to remove PII
        safe_details = self._sanitize_details(details or {})

        audit_entry = AuditLog(
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            org_id=org_id,
            action=action,
            details=safe_details,
            client_ip=request_context.get("client_ip") if request_context else None,
            user_agent=request_context.get("user_agent") if request_context else None,
            timestamp=datetime.utcnow()
        )

        db.add(audit_entry)
        db.commit()

        # Also log to structured logger for ELK
        logger.info(
            "audit_event",
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            user_id=user_id
        )

    def _sanitize_details(self, details: dict) -> dict:
        """Remove sensitive information from audit details."""
        sanitized = {}
        for key, value in details.items():
            if isinstance(value, str):
                sanitized[key] = self.pii_protector.mask_pii_for_logging(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = value
        return sanitized

    def log_document_upload(self, document_id: str, user_id: str, org_id: str, metadata: dict):
        """Log document upload event."""
        self.log_event(
            event_type="DOCUMENT_UPLOAD",
            resource_type="document",
            resource_id=document_id,
            action="CREATE",
            user_id=user_id,
            org_id=org_id,
            details={
                "filename": metadata.get("filename"),
                "category": metadata.get("category"),
                "file_size": metadata.get("file_size")
            }
        )

    def log_query(self, query_id: str, user_id: str, org_id: str, question: str, sources: list):
        """Log query event."""
        self.log_event(
            event_type="POLICY_QUERY",
            resource_type="query",
            resource_id=query_id,
            action="READ",
            user_id=user_id,
            org_id=org_id,
            details={
                "question_length": len(question),
                "sources_accessed": [str(s["document_id"]) for s in sources],
                "source_count": len(sources)
            }
        )

    def log_document_access(self, document_id: str, user_id: str, org_id: str, access_type: str):
        """Log document access for compliance."""
        self.log_event(
            event_type="DOCUMENT_ACCESS",
            resource_type="document",
            resource_id=document_id,
            action=access_type,
            user_id=user_id,
            org_id=org_id
        )
```

### 4.5 Compliance Considerations

#### GDPR Compliance

```python
class GDPRCompliance:
    def __init__(self):
        self.data_retention_days = 365
        self.audit_retention_days = 2555  # 7 years

    def handle_data_subject_request(self, user_id: str, request_type: str):
        """Handle GDPR data subject requests."""
        if request_type == "access":
            return self._export_user_data(user_id)
        elif request_type == "deletion":
            return self._delete_user_data(user_id)
        elif request_type == "portability":
            return self._export_portable_data(user_id)
        elif request_type == "rectification":
            return self._flag_for_rectification(user_id)

    def _export_user_data(self, user_id: str) -> dict:
        """Export all data associated with user (GDPR Article 15)."""
        return {
            "user_profile": db.query(User).filter(User.id == user_id).first(),
            "query_history": db.query(QueryLog).filter(QueryLog.user_id == user_id).all(),
            "documents_uploaded": db.query(Document).filter(Document.uploaded_by == user_id).all(),
            "audit_logs": db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
        }

    def _delete_user_data(self, user_id: str) -> dict:
        """Delete user data (GDPR Article 17 - Right to Erasure)."""
        # Anonymize rather than delete for audit trail
        queries_updated = db.query(QueryLog).filter(
            QueryLog.user_id == user_id
        ).update({"user_id": f"DELETED_{user_id[:8]}"})

        # Remove personally uploaded documents
        documents_deleted = db.query(Document).filter(
            Document.uploaded_by == user_id
        ).delete()

        return {
            "queries_anonymized": queries_updated,
            "documents_deleted": documents_deleted,
            "deletion_timestamp": datetime.utcnow().isoformat()
        }

    def enforce_data_retention(self):
        """Automatically purge data beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.data_retention_days)

        # Delete old query logs
        old_queries = db.query(QueryLog).filter(
            QueryLog.created_at < cutoff_date
        ).delete()

        # Archive old audit logs (keep for compliance but move to cold storage)
        audit_cutoff = datetime.utcnow() - timedelta(days=self.audit_retention_days)
        old_audits = db.query(AuditLog).filter(
            AuditLog.timestamp < audit_cutoff
        ).delete()

        logger.info(
            "data_retention_enforced",
            queries_deleted=old_queries,
            audits_archived=old_audits
        )

    def get_consent_status(self, user_id: str) -> dict:
        """Check user consent status."""
        user = db.query(User).filter(User.id == user_id).first()
        return {
            "data_processing_consent": user.consent_data_processing,
            "analytics_consent": user.consent_analytics,
            "consent_date": user.consent_date,
            "consent_version": user.consent_version
        }
```

---

## Document 5: Deployment & Infrastructure

### 5.1 Docker Containerization

```dockerfile
# Dockerfile for main API service
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```dockerfile
# Dockerfile for document processing worker
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r worker && useradd -r -g worker worker

WORKDIR /app

COPY requirements-worker.txt .
RUN pip install -r requirements-worker.txt

COPY . .

RUN chown -R worker:worker /app
USER worker

CMD ["python", "-m", "workers.document_worker"]
```

```yaml
# docker-compose.yml for local development
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/hr_assistant
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - ENVIRONMENT=development
    depends_on:
      - db
      - redis
    volumes:
      - ./:/app
    networks:
      - hr_network

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/hr_assistant
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - S3_ENDPOINT=http://minio:9000
    depends_on:
      - db
      - redis
      - minio
    networks:
      - hr_network

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=hr_assistant
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - hr_network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - hr_network

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    networks:
      - hr_network

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - hr_network

volumes:
  postgres_data:
  redis_data:
  minio_data:
  qdrant_data:

networks:
  hr_network:
    driver: bridge
```

### 5.2 Environment Configuration

```python
# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "HR Assistant RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

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

    # Vector Store
    VECTOR_STORE_TYPE: str = "qdrant"  # or "faiss"
    QDRANT_URL: Optional[str] = None
    QDRANT_COLLECTION: str = "hr_policies"
    FAISS_INDEX_PATH: Optional[str] = None

    # LLM Configuration
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Authentication
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 1

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    DEFAULT_RATE_LIMIT: int = 60  # per minute

    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: list = [".pdf", ".docx", ".txt", ".html", ".md"]
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Security
    ENCRYPTION_KEY: str
    CORS_ORIGINS: list = ["*"]
    ENABLE_PII_DETECTION: bool = True

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

DATABASE_URL=postgresql://postgres:password@localhost:5432/hr_assistant
REDIS_URL=redis://localhost:6379/0

S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

QDRANT_URL=http://localhost:6333

OPENAI_API_KEY=sk-your-dev-key
JWT_SECRET=dev-secret-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key

RATE_LIMIT_ENABLED=false
```

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

DATABASE_URL=postgresql://prod_user:${DB_PASSWORD}@prod-db.internal:5432/hr_assistant
DATABASE_POOL_SIZE=50
REDIS_URL=redis://prod-redis.internal:6379/0

S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=${AWS_ACCESS_KEY}
S3_SECRET_KEY=${AWS_SECRET_KEY}
S3_BUCKET_NAME=hr-assistant-documents-prod

QDRANT_URL=http://qdrant.internal:6333

OPENAI_API_KEY=${OPENAI_API_KEY}
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

RATE_LIMIT_ENABLED=true
DEFAULT_RATE_LIMIT=100

PROMETHEUS_ENABLED=true
```

### 5.3 CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
          black app/ --check
          mypy app/

      - name: Run security checks
        run: |
          bandit -r app/
          safety check

      - name: Run tests
        env:
          DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
          OPENAI_API_KEY: test-key
          JWT_SECRET: test-secret
        run: |
          pytest tests/ -v --cov=app --cov-report=xml --cov-report=html

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha

      - name: Build and push API image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}-api
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build and push Worker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.worker
          push: true
          tags: ${{ steps.meta.outputs.tags }}-worker
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
      - name: Deploy to Staging
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/hr-assistant
            docker-compose pull
            docker-compose up -d
            docker-compose exec -T api python -m alembic upgrade head

      - name: Run smoke tests
        run: |
          curl -f https://staging-api.hr-assistant.com/health || exit 1

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
      - name: Deploy to Production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            kubectl set image deployment/hr-assistant-api \
              api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}-api
            kubectl set image deployment/hr-assistant-worker \
              worker=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}-worker
            kubectl rollout status deployment/hr-assistant-api
            kubectl rollout status deployment/hr-assistant-worker

      - name: Verify deployment
        run: |
          sleep 30
          curl -f https://api.hr-assistant.com/health || exit 1

      - name: Notify team
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "HR Assistant deployed to production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "✅ *HR Assistant* deployed to production\nCommit: ${{ github.sha }}\nBy: ${{ github.actor }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 5.4 Monitoring, Logging, and Alerting

```yaml
# kubernetes/monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: 'hr-assistant-api'
        static_configs:
          - targets: ['hr-assistant-api:8000']
        metrics_path: '/metrics'

      - job_name: 'hr-assistant-worker'
        static_configs:
          - targets: ['hr-assistant-worker:9090']

    alerting:
      alertmanagers:
        - static_configs:
            - targets: ['alertmanager:9093']

    rule_files:
      - /etc/prometheus/alerts.yml

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
data:
  alerts.yml: |
    groups:
      - name: hr_assistant_alerts
        rules:
          - alert: HighQueryLatency
            expr: histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m])) > 5
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High query latency detected"
              description: "95th percentile query latency is above 5 seconds"

          - alert: HighErrorRate
            expr: rate(rag_queries_total{status="error"}[5m]) / rate(rag_queries_total[5m]) > 0.01
            for: 2m
            labels:
              severity: critical
            annotations:
              summary: "High error rate in RAG queries"
              description: "Error rate is above 1%"

          - alert: LLMServiceDown
            expr: up{job="openai-health"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "LLM service is unavailable"

          - alert: LowDiskSpace
            expr: node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes < 0.1
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "Low disk space on data volume"

          - alert: HighMemoryUsage
            expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High memory usage detected"
```

```python
# Structured logging configuration
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
            "foreign_pre_chain": [
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "/var/log/hr-assistant/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### 5.5 Database and Vector Store Persistence

```yaml
# kubernetes/persistence.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: hr_assistant
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
  volumeClaimTemplates:
    - metadata:
        name: postgres-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
          resources:
            requests:
              memory: "8Gi"
              cpu: "4"
            limits:
              memory: "16Gi"
              cpu: "8"
      volumes:
        - name: qdrant-storage
          persistentVolumeClaim:
            claimName: qdrant-pvc
```

---

## Document 6: Error Handling & Observability

### 6.1 Exception Hierarchy

```python
# exceptions.py
class HRAssistantError(Exception):
    """Base exception for HR Assistant."""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)

# Authentication errors
class AuthenticationError(HRAssistantError):
    """Authentication failed."""
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, "AUTHENTICATION_FAILED", details)

class AuthorizationError(HRAssistantError):
    """Insufficient permissions."""
    def __init__(self, message: str = "Insufficient permissions", details: dict = None):
        super().__init__(message, "AUTHORIZATION_FAILED", details)

# Document processing errors
class DocumentProcessingError(HRAssistantError):
    """Document processing failed."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", details)

class UnsupportedFileTypeError(DocumentProcessingError):
    """File type not supported."""
    def __init__(self, file_type: str):
        super().__init__(
            f"File type '{file_type}' is not supported",
            {"supported_types": [".pdf", ".docx", ".txt", ".html", ".md"]}
        )
        self.error_code = "UNSUPPORTED_FILE_TYPE"

class FileTooLargeError(DocumentProcessingError):
    """File exceeds size limit."""
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            f"File size {file_size} bytes exceeds maximum {max_size} bytes",
            {"file_size": file_size, "max_size": max_size}
        )
        self.error_code = "FILE_TOO_LARGE"

# RAG pipeline errors
class RAGPipelineError(HRAssistantError):
    """RAG pipeline error."""
    pass

class EmbeddingError(RAGPipelineError):
    """Embedding generation failed."""
    def __init__(self, message: str = "Failed to generate embeddings", details: dict = None):
        super().__init__(message, "EMBEDDING_ERROR", details)

class VectorSearchError(RAGPipelineError):
    """Vector search failed."""
    def __init__(self, message: str = "Vector search failed", details: dict = None):
        super().__init__(message, "VECTOR_SEARCH_ERROR", details)

class LLMError(RAGPipelineError):
    """LLM service error."""
    def __init__(self, message: str = "LLM service error", details: dict = None):
        super().__init__(message, "LLM_ERROR", details)

class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""
    def __init__(self, retry_after: int = None):
        super().__init__(
            "LLM rate limit exceeded",
            {"retry_after_seconds": retry_after}
        )
        self.error_code = "LLM_RATE_LIMIT"

class LLMTimeoutError(LLMError):
    """LLM request timeout."""
    def __init__(self, timeout_seconds: int):
        super().__init__(
            f"LLM request timed out after {timeout_seconds} seconds",
            {"timeout_seconds": timeout_seconds}
        )
        self.error_code = "LLM_TIMEOUT"

# Validation errors
class ValidationError(HRAssistantError):
    """Input validation failed."""
    def __init__(self, field: str, issue: str):
        super().__init__(
            f"Validation failed for field '{field}': {issue}",
            "VALIDATION_ERROR",
            {"field": field, "issue": issue}
        )

# Resource errors
class ResourceNotFoundError(HRAssistantError):
    """Resource not found."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} with ID '{resource_id}' not found",
            "RESOURCE_NOT_FOUND",
            {"resource_type": resource_type, "resource_id": resource_id}
        )

# Rate limiting
class RateLimitExceededError(HRAssistantError):
    """Rate limit exceeded."""
    def __init__(self, limit: int, window: str, retry_after: int):
        super().__init__(
            f"Rate limit of {limit} requests per {window} exceeded",
            "RATE_LIMIT_EXCEEDED",
            {
                "limit": limit,
                "window": window,
                "retry_after_seconds": retry_after
            }
        )
```

### 6.2 Retry Logic and Circuit Breaker

```python
import time
from functools import wraps
from typing import Callable, Type, Tuple
import random

class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Reset failure count on success."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Increment failure count and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e)
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e)
                    )

                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


# Usage example
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=LLMError
)

class RobustLLMService:
    def __init__(self):
        self.client = OpenAI()
        self.circuit_breaker = llm_circuit_breaker

    @retry_with_exponential_backoff(
        max_retries=3,
        retryable_exceptions=(LLMRateLimitError, LLMTimeoutError)
    )
    def generate_response(self, prompt: str) -> str:
        """Generate LLM response with retry and circuit breaker."""
        def _call():
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                raise LLMRateLimitError(retry_after=60)
            except openai.APITimeoutError:
                raise LLMTimeoutError(timeout_seconds=30)
            except Exception as e:
                raise LLMError(str(e))

        return self.circuit_breaker.call(_call)
```

### 6.3 Structured Logging Strategy

```python
import structlog
from contextvars import ContextVar
import uuid

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")

def add_request_context(logger, method_name, event_dict):
    """Add request context to all log entries."""
    event_dict["request_id"] = request_id_var.get()
    event_dict["user_id"] = user_id_var.get()
    return event_dict

logger = structlog.get_logger()

# Logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Add structured logging to all requests."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    # Extract user ID from token if available
    if hasattr(request.state, "user"):
        user_id_var.set(request.state.user.id)

    # Log request start
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        query_params=str(request.query_params),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent", "")
    )

    start_time = time.time()

    try:
        response = await call_next(request)

        # Log request completion
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=int((time.time() - start_time) * 1000)
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        # Log request failure
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=int((time.time() - start_time) * 1000),
            exc_info=True
        )
        raise


# Domain-specific logging
class QueryLogger:
    """Structured logging for RAG queries."""

    @staticmethod
    def log_query_start(query_id: str, question: str, filters: dict):
        logger.info(
            "rag_query_started",
            query_id=query_id,
            question_length=len(question),
            filters=filters
        )

    @staticmethod
    def log_retrieval(query_id: str, chunks_retrieved: int, avg_score: float):
        logger.info(
            "rag_retrieval_completed",
            query_id=query_id,
            chunks_retrieved=chunks_retrieved,
            average_relevance_score=avg_score
        )

    @staticmethod
    def log_llm_call(query_id: str, tokens_used: int, latency_ms: int):
        logger.info(
            "rag_llm_completed",
            query_id=query_id,
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )

    @staticmethod
    def log_query_complete(
        query_id: str,
        confidence: float,
        total_latency_ms: int,
        cached: bool
    ):
        logger.info(
            "rag_query_completed",
            query_id=query_id,
            confidence=confidence,
            total_latency_ms=total_latency_ms,
            cached=cached
        )

    @staticmethod
    def log_query_error(query_id: str, error: Exception):
        logger.error(
            "rag_query_failed",
            query_id=query_id,
            error=str(error),
            error_type=type(error).__name__,
            exc_info=True
        )
```

### 6.4 Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# RAG pipeline metrics
RAG_QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status", "cached"]
)

RAG_QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query end-to-end latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Vector retrieval latency",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)

LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

RETRIEVAL_SCORE = Summary(
    "rag_retrieval_relevance_score",
    "Relevance scores of retrieved documents"
)

CONFIDENCE_SCORE = Summary(
    "rag_response_confidence",
    "Confidence scores of RAG responses"
)

# Token usage metrics
TOKENS_USED = Counter(
    "rag_tokens_total",
    "Total tokens consumed",
    ["type"]  # prompt, completion
)

ESTIMATED_COST = Counter(
    "rag_estimated_cost_usd",
    "Estimated cost in USD"
)

# Document metrics
DOCUMENTS_INDEXED = Gauge(
    "rag_documents_indexed_total",
    "Total number of indexed documents"
)

CHUNKS_INDEXED = Gauge(
    "rag_chunks_indexed_total",
    "Total number of indexed chunks"
)

VECTOR_STORE_SIZE = Gauge(
    "rag_vector_store_size_bytes",
    "Size of vector store in bytes"
)

# Processing metrics
DOCUMENT_PROCESSING_TIME = Histogram(
    "rag_document_processing_seconds",
    "Document processing time",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

DOCUMENT_PROCESSING_ERRORS = Counter(
    "rag_document_processing_errors_total",
    "Document processing errors",
    ["error_type"]
)

# System health metrics
CACHE_HIT_RATE = Gauge(
    "rag_cache_hit_rate",
    "Cache hit rate percentage"
)

CIRCUIT_BREAKER_STATE = Gauge(
    "rag_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["service"]
)

# Business metrics
QUERIES_PER_USER = Counter(
    "rag_queries_per_user_total",
    "Queries per user",
    ["user_id"]
)

POPULAR_CATEGORIES = Counter(
    "rag_query_categories_total",
    "Query categories accessed",
    ["category"]
)


# Metrics collection in pipeline
class MetricsCollector:
    @staticmethod
    def record_query(
        status: str,
        cached: bool,
        latency: float,
        confidence: float,
        tokens_used: dict
    ):
        RAG_QUERY_COUNT.labels(status=status, cached=str(cached).lower()).inc()
        RAG_QUERY_LATENCY.observe(latency)
        CONFIDENCE_SCORE.observe(confidence)

        TOKENS_USED.labels(type="prompt").inc(tokens_used["prompt"])
        TOKENS_USED.labels(type="completion").inc(tokens_used["completion"])

        # Calculate cost (GPT-4o-mini pricing)
        cost = (
            tokens_used["prompt"] * 0.00015 / 1000 +
            tokens_used["completion"] * 0.0006 / 1000
        )
        ESTIMATED_COST.inc(cost)

    @staticmethod
    def record_retrieval(num_chunks: int, scores: list, latency: float):
        RETRIEVAL_LATENCY.observe(latency)
        for score in scores:
            RETRIEVAL_SCORE.observe(score)

    @staticmethod
    def update_system_metrics(documents: int, chunks: int, vector_size: int):
        DOCUMENTS_INDEXED.set(documents)
        CHUNKS_INDEXED.set(chunks)
        VECTOR_STORE_SIZE.set(vector_size)
```

### 6.5 Health Check Endpoints

```python
from datetime import datetime
import asyncio

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with dependency status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "checks": {}
    }

    # Check database
    db_status = await check_database_health()
    health_status["checks"]["database"] = db_status

    # Check Redis
    redis_status = await check_redis_health()
    health_status["checks"]["redis"] = redis_status

    # Check vector store
    vector_status = await check_vector_store_health()
    health_status["checks"]["vector_store"] = vector_status

    # Check LLM service
    llm_status = await check_llm_health()
    health_status["checks"]["llm_service"] = llm_status

    # Check object storage
    storage_status = await check_storage_health()
    health_status["checks"]["object_storage"] = storage_status

    # Determine overall status
    all_checks = health_status["checks"].values()
    if any(check["status"] == "unhealthy" for check in all_checks):
        health_status["status"] = "unhealthy"
    elif any(check["status"] == "degraded" for check in all_checks):
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


async def check_database_health() -> dict:
    """Check PostgreSQL connectivity and performance."""
    try:
        start = time.time()
        result = await db.execute("SELECT 1")
        latency = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "details": {"connection_pool_size": db.pool.size()}
        }
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_redis_health() -> dict:
    """Check Redis connectivity."""
    try:
        start = time.time()
        await redis_client.ping()
        latency = (time.time() - start) * 1000

        info = await redis_client.info()
        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "details": {
                "used_memory_mb": info["used_memory"] / 1024 / 1024,
                "connected_clients": info["connected_clients"]
            }
        }
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_vector_store_health() -> dict:
    """Check vector store status."""
    try:
        start = time.time()
        # Perform a simple search to verify functionality
        test_vector = [0.0] * 1536
        results = await vector_store.search(test_vector, k=1)
        latency = (time.time() - start) * 1000

        collection_info = await vector_store.get_collection_info()
        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "details": {
                "vectors_count": collection_info["vectors_count"],
                "index_size_mb": collection_info["size_mb"]
            }
        }
    except Exception as e:
        logger.error("vector_store_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_llm_health() -> dict:
    """Check LLM service availability."""
    try:
        # Check circuit breaker state
        if llm_circuit_breaker.state == "OPEN":
            return {
                "status": "degraded",
                "details": {
                    "circuit_breaker": "OPEN",
                    "recovery_in_seconds": max(
                        0,
                        llm_circuit_breaker.recovery_timeout -
                        (time.time() - llm_circuit_breaker.last_failure_time)
                    )
                }
            }

        # Perform lightweight API check
        start = time.time()
        response = await openai_client.models.list()
        latency = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "details": {
                "circuit_breaker": llm_circuit_breaker.state,
                "failure_count": llm_circuit_breaker.failure_count
            }
        }
    except Exception as e:
        logger.error("llm_health_check_failed", error=str(e))
        return {
            "status": "degraded",
            "error": str(e)
        }


async def check_storage_health() -> dict:
    """Check object storage connectivity."""
    try:
        start = time.time()
        # List buckets as a health check
        await s3_client.list_buckets()
        latency = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        logger.error("storage_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to serve traffic
    try:
        # Quick checks for critical dependencies
        await db.execute("SELECT 1")
        await redis_client.ping()

        return {"status": "ready"}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        return JSONResponse(
            content={"status": "not ready", "error": str(e)},
            status_code=503
        )


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Simple check that application is running
    return {"status": "alive"}
```

---

## Summary

This comprehensive architecture documentation provides a production-ready foundation for building an enterprise-grade HR Assistant with RAG capabilities. The key architectural decisions prioritize:

1. **Scalability**: Stateless API design, horizontal scaling, sharded vector stores
2. **Security**: Multi-layer authentication, RBAC, encryption, audit logging
3. **Reliability**: Circuit breakers, retry logic, health checks, graceful degradation
4. **Observability**: Structured logging, comprehensive metrics, distributed tracing
5. **Compliance**: GDPR-ready data handling, audit trails, data retention policies
6. **Cost Efficiency**: Caching strategies, model routing, token budget management

The modular design allows teams to start small and scale progressively based on organizational needs and budget constraints.
