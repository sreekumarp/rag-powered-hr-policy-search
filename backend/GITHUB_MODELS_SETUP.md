# GitHub Models Integration - Setup Guide

This guide will help you integrate GitHub Models (GPT-4o-mini) with your HR RAG system.

## Quick Start

### 1. Get GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. **No permissions needed** - leave all checkboxes unchecked
4. Generate token and copy it

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your GitHub token:

```bash
# Add your GitHub token
GITHUB_TOKEN=ghp_your_actual_token_here

# Enable LLM
LLM_ENABLED=true

# Model settings (defaults are fine)
GITHUB_MODELS_MODEL=gpt-4o-mini
GITHUB_MODELS_MAX_TOKENS=500
GITHUB_MODELS_TEMPERATURE=0.1

# Caching (important!)
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=86400
LLM_CACHE_SIMILARITY=0.92

# Features
LLM_ENABLE_CITATIONS=true
LLM_FALLBACK_TO_TEMPLATE=true
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The new dependencies added are:
- `openai==1.12.0` - GitHub Models uses OpenAI-compatible API
- `tiktoken==0.6.0` - Token counting

### 4. Run Tests

```bash
python test_github_models.py
```

### 5. Start Application

```bash
python app.py
```

## Features

### 🤖 LLM-Powered Responses

Queries now use GPT-4o-mini for natural language generation with:
- Proper context synthesis
- Source citations ([Source 1], [Source 2], etc.)
- Professional HR tone
- Grounded in retrieved documents

### 💾 Semantic Caching

Automatic caching to stay within free tier limits (50 requests/day):
- Semantic similarity matching (not exact match)
- 40-50% expected cache hit rate
- 24-hour TTL
- Persisted to disk

### 🔄 Automatic Fallback

If rate limited or GitHub Models unavailable:
- Automatically falls back to template-based responses
- User still gets an answer
- No errors or failures

### 📊 Admin Monitoring

New admin endpoints:

**Get LLM Statistics:**
```bash
GET /api/admin/llm/stats
```

Response:
```json
{
  "llm": {
    "total_requests": 45,
    "github_requests": 22,
    "template_requests": 3,
    "cache_hits": 20,
    "github_daily_limit": 50,
    "github_requests_today": 22,
    "github_remaining_today": 28
  },
  "cache": {
    "enabled": true,
    "hits": 20,
    "misses": 25,
    "hit_rate": 0.44,
    "size": 25
  }
}
```

**Clear Cache:**
```bash
POST /api/admin/llm/cache/clear
```

**Test LLM Connection:**
```bash
POST /api/admin/llm/test
{
  "test_query": "Say hello"
}
```

## Rate Limits (GitHub Models Free Tier)

- **GPT-4o-mini**: 50 requests/day
- **Per minute**: 10 requests/minute
- **Concurrent**: 2 max

### Expected Daily Capacity

With 50 requests/day and 40% cache hit rate:
- **Effective capacity**: ~83 unique queries/day
- If exceeded, automatic fallback to templates

## Query Examples

### With LLM (GPT-4o-mini)

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many vacation days do full-time employees get?"
  }'
```

Response:
```json
{
  "answer": "According to [Source 1], full-time employees receive 15 days of paid vacation per year.",
  "confidence": 0.87,
  "llm_metadata": {
    "provider": "github-models",
    "model": "gpt-4o-mini",
    "latency_ms": 1243,
    "cached": false
  },
  "sources": [...],
  "processing_time_ms": 1350
}
```

### Cached Response

```json
{
  "answer": "According to [Source 1], full-time employees receive 15 days of paid vacation per year.",
  "confidence": 0.87,
  "llm_metadata": {
    "provider": "github-models",
    "model": "gpt-4o-mini",
    "latency_ms": 82,
    "cached": true,
    "cache_similarity": 0.95
  },
  "processing_time_ms": 145
}
```

## Troubleshooting

### Issue: "Authentication failed - check GITHUB_TOKEN"

**Solution:**
1. Verify token is set: `echo $GITHUB_TOKEN`
2. Token should start with `ghp_`
3. Regenerate token if needed
4. No permissions required for token

### Issue: "Rate limit exceeded"

**Solution:**
- Check admin stats: `GET /api/admin/llm/stats`
- Wait until next day (resets every 24h)
- System automatically falls back to template
- Clear cache to free up old entries: `POST /api/admin/llm/cache/clear`

### Issue: Responses are template-based, not LLM

**Solution:**
1. Check `LLM_ENABLED=true` in .env
2. Verify GitHub token is valid
3. Check logs for error messages
4. Test LLM: `POST /api/admin/llm/test`

### Issue: Cache not working

**Solution:**
1. Verify `LLM_CACHE_ENABLED=true`
2. Check cache file exists: `ls data/llm_cache.json`
3. Lower similarity threshold: `LLM_CACHE_SIMILARITY=0.90`
4. Check cache stats: `GET /api/admin/llm/stats`

## Performance

### Latency

- **With Cache Hit**: 50-150ms
- **With GitHub Models**: 800-2000ms
- **With Template**: 100-200ms

### Quality

- **GitHub Models (GPT-4o-mini)**: High-quality synthesis with citations
- **Template Fallback**: Basic text extraction, lower quality

## Cost

**GitHub Models (Free Tier)**: $0 (FREE!)
- Limit: 50 requests/day
- Model: GPT-4o-mini (same quality as OpenAI)

**Comparison to Paid Options:**

| Provider | Model | Cost/1K Queries | Daily Limit |
|----------|-------|-----------------|-------------|
| **GitHub Models** | GPT-4o-mini | **$0** | 50 (free) |
| OpenAI | GPT-4o-mini | $0.15 | Unlimited* |
| Anthropic | Claude-3-Haiku | $0.13 | Unlimited* |

*Subject to billing limits

## Next Steps

1. **Monitor Usage**
   - Check `/api/admin/llm/stats` daily
   - Track cache hit rate
   - Observe query patterns

2. **Optimize Caching**
   - Tune similarity threshold based on hit rate
   - Increase TTL for stable policies
   - Decrease TTL for frequently updated policies

3. **Gather Feedback**
   - Compare LLM responses vs template
   - Collect user ratings
   - Identify low-confidence queries

4. **Scale If Needed**
   - If >50 queries/day, consider:
     - Paid GitHub Models tier
     - OpenAI API (BYOK)
     - Local Ollama instance

## Architecture

The integration adds these new components:

1. **LLMService** ([services/llm_service.py](services/llm_service.py))
   - GitHub Models provider
   - Template fallback
   - Rate limit tracking
   - Usage statistics

2. **ResponseCache** ([utils/cache.py](utils/cache.py))
   - Semantic caching
   - TTL-based expiration
   - Disk persistence
   - LRU eviction

3. **PromptBuilder** ([utils/prompts.py](utils/prompts.py))
   - Optimized prompts for HR queries
   - Source citation instructions
   - Professional tone

4. **RAGPipeline Updates** ([utils/rag_pipeline.py](utils/rag_pipeline.py))
   - LLM integration
   - Cache checking
   - Automatic fallback

## References

- [GitHub Models Documentation](https://docs.github.com/en/github-models)
- [Using GPT-4 for Free Through Github Models](https://2coffee.dev/en/articles/using-gpt-4-for-free-with-github-models)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
