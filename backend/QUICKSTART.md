# Quick Start Guide - HR Assistant RAG Enterprise Edition

This guide will help you get the HR Assistant RAG system up and running in minutes.

## Prerequisites

- Python 3.11+
- Virtual environment support
- (Optional) Docker and Docker Compose for containerized deployment
- (Optional) PostgreSQL for production database

## Option 1: Local Development (Fastest)

### Step 1: Setup Environment

```bash
# Navigate to the project directory
cd backend

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Fix huggingface_hub compatibility
pip install 'huggingface_hub<0.17'

# Install CLI tool
pip install click
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for development)
# The defaults work fine for local development
```

### Step 4: Initialize Database

```bash
# Initialize database and create admin user
python manage.py init-db

# Create admin user (interactive)
python manage.py create-admin
# Or use defaults: username=admin, password=admin123
```

### Step 5: Start the Application

```bash
# Run on port 51425 (or 5001 if 51425 is in use on macOS)
export PORT=5001  # Optional
python app.py
```

The API will be available at: `http://localhost:5001`

### Step 6: Test the API

```bash
# Health check
curl http://localhost:5001/api/health

# API info
curl http://localhost:5001/api/info

# Register a user
curl -X POST http://localhost:5001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'

# Login
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'

# Save the access_token from the response
export TOKEN="your-access-token-here"

# Index sample documents
curl -X POST http://localhost:5001/api/documents/index-sample \
  -H "Authorization: Bearer $TOKEN"

# Query the system
curl -X POST http://localhost:5001/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many vacation days do I get?"
  }'
```

## Option 2: Docker Compose (Production-Ready)

### Step 1: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# IMPORTANT: Edit .env and set secure passwords!
# - SECRET_KEY
# - JWT_SECRET_KEY
# - POSTGRES_PASSWORD
# - ADMIN_PASSWORD
```

### Step 2: Build and Run

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f api
```

The API will be available at: `http://localhost:51425`

Services included:
- API server on port 51425
- PostgreSQL on port 5432
- Redis on port 6379

### Step 3: Initialize Database

```bash
# Access the container
docker-compose exec api bash

# Create admin user
python manage.py create-admin

# Exit container
exit
```

### Step 4: Test the Deployment

```bash
# Health check
curl http://localhost:51425/api/health

# Check logs
docker-compose logs -f api
```

## Common Commands

### User Management

```bash
# List all users
python manage.py list-users

# Create a new user
python manage.py create-user john john@example.com password123 --role user

# Change user role
python manage.py change-role 2 admin

# Deactivate user
python manage.py deactivate 2

# View statistics
python manage.py stats

# Create API key for a user
python manage.py create-api-key testuser
```

### Database Management

```bash
# Initialize database
python manage.py init-db

# Reset database (CAUTION: Deletes all data!)
python manage.py reset-db

# Using Flask-Migrate for migrations
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Restart a service
docker-compose restart api

# Execute commands in container
docker-compose exec api python manage.py list-users

# Clean up (remove volumes - DELETES ALL DATA!)
docker-compose down -v
```

## API Endpoints Overview

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get tokens
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user profile
- `POST /api/auth/change-password` - Change password
- `GET /api/auth/api-keys` - List API keys
- `POST /api/auth/api-keys` - Create API key
- `DELETE /api/auth/api-keys/<id>` - Revoke API key

### Documents
- `GET /api/documents` - List documents
- `GET /api/documents/<source_id>` - Get document details
- `POST /api/documents` - Upload document
- `DELETE /api/documents/<source_id>` - Delete document
- `PATCH /api/documents/<source_id>` - Update metadata
- `GET /api/documents/statistics` - Document stats
- `POST /api/documents/index-sample` - Index sample FAQ

### Query & Analytics
- `POST /api/query` - Query HR policies
- `GET /api/query/history` - Query history
- `GET /api/analytics` - Query analytics
- `GET /api/analytics/popular` - Popular queries
- `GET /api/analytics/low-confidence` - Low confidence queries

### Admin (Admin role required)
- `GET /api/admin/users` - List all users
- `GET /api/admin/users/<id>` - Get user details
- `PATCH /api/admin/users/<id>/role` - Update user role
- `POST /api/admin/users/<id>/deactivate` - Deactivate user
- `GET /api/admin/stats` - System statistics
- `DELETE /api/admin/clear-data` - Clear all data
- `GET /api/admin/logs/queries` - Query logs
- `GET /api/admin/config` - System configuration

### Public
- `GET /api/health` - Health check (no auth)
- `GET /api/info` - API information (no auth)

## Authentication Methods

### 1. JWT Token (Recommended for web/mobile apps)

```bash
# Login to get token
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "password123"}'

# Use token in requests
curl -X POST http://localhost:5001/api/query \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days?"}'
```

### 2. API Key (Recommended for scripts/automation)

```bash
# Create API key
python manage.py create-api-key testuser

# Use API key in requests
curl -X POST http://localhost:5001/api/query \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days?"}'
```

## Next Steps

1. **Upload Your Documents**: Use `POST /api/documents` to upload PDF, DOCX, TXT files
2. **Configure Permissions**: Set up users with appropriate roles (readonly, user, admin)
3. **Monitor Usage**: Check `/api/analytics` for query statistics
4. **Customize**: Edit `config.py` to adjust chunk sizes, confidence thresholds, etc.
5. **Scale**: Use PostgreSQL + Redis for production deployments

## Troubleshooting

### Port 51425 in use (macOS)
```bash
# AirPlay uses port 51425 on macOS
export PORT=5001
python app.py
```

### huggingface_hub ImportError
```bash
pip install 'huggingface_hub<0.17'
```

### Database locked (SQLite)
```bash
# Use PostgreSQL for production or ensure single process access
```

### Rate limit errors
```bash
# Disable in development
# Edit .env: RATE_LIMIT_ENABLED=false
```

## Support

For issues or questions:
- Check the logs in `logs/app.log`
- Review the full documentation in `README.md`
- Use `python manage.py stats` to check system health
