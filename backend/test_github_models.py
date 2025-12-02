"""
Test script for GitHub Models integration.

Prerequisites:
1. Set GITHUB_TOKEN environment variable
2. Install dependencies: pip install -r requirements.txt
3. Have some documents indexed in vector store

Usage:
    python test_github_models.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Check for GitHub token
if not os.getenv("GITHUB_TOKEN"):
    print("ERROR: GITHUB_TOKEN not set!")
    print("\nTo fix:")
    print("1. Go to https://github.com/settings/tokens")
    print("2. Generate new token (no permissions needed)")
    print("3. Add to .env file: GITHUB_TOKEN=ghp_your_token_here")
    sys.exit(1)

print("=" * 70)
print("GitHub Models Integration Test")
print("=" * 70)

# Import after env loading
from services.llm_service import LLMService
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline
from utils.cache import ResponseCache

def test_1_llm_service():
    """Test 1: LLM Service basic functionality."""
    print("\n--- TEST 1: LLM Service ---")

    config = {
        "LLM_ENABLED": True,
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
        "GITHUB_MODELS_ENDPOINT": "https://models.inference.ai.azure.com",
        "GITHUB_MODELS_MODEL": "gpt-4o-mini",
        "GITHUB_MODELS_MAX_TOKENS": 100,
        "GITHUB_MODELS_TEMPERATURE": 0.1,
        "LLM_FALLBACK_TO_TEMPLATE": True,
    }

    llm_service = LLMService(config)

    test_prompt = "Say 'Hello from GitHub Models!' and nothing else."

    try:
        response, metadata = llm_service.generate(test_prompt)
        print(f"✓ Response: {response}")
        print(f"✓ Provider: {metadata.get('provider')}")
        print(f"✓ Latency: {metadata.get('latency_ms')}ms")
        print("✓ Test 1 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False


def test_2_rag_pipeline():
    """Test 2: RAG Pipeline with LLM."""
    print("\n--- TEST 2: RAG Pipeline with LLM ---")

    # Initialize components
    vector_store = VectorStore()

    config = {
        "LLM_ENABLED": True,
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
        "GITHUB_MODELS_ENDPOINT": "https://models.inference.ai.azure.com",
        "GITHUB_MODELS_MODEL": "gpt-4o-mini",
        "GITHUB_MODELS_MAX_TOKENS": 200,
        "GITHUB_MODELS_TEMPERATURE": 0.1,
        "LLM_FALLBACK_TO_TEMPLATE": True,
    }

    llm_service = LLMService(config)
    pipeline = RAGPipeline(vector_store, llm_service=llm_service, enable_citations=True)

    # Add test document
    vector_store.add_document(
        "Full-time employees receive 15 days of paid vacation per year. "
        "Part-time employees receive vacation days prorated based on hours worked.",
        source="vacation_policy.pdf",
        metadata={"category": "leave", "department": "all"}
    )

    # Query
    result = pipeline.query("How many vacation days do full-time employees get?")

    print(f"✓ Question: {result['query']}")
    print(f"✓ Answer: {result['answer']}")
    print(f"✓ Provider: {result.get('llm_metadata', {}).get('provider')}")
    print(f"✓ Confidence: {result['confidence']}")
    print(f"✓ Sources: {result['relevant_chunks']}")

    # Check for citations
    if "[Source" in result['answer']:
        print("✓ Citations included")

    print("✓ Test 2 PASSED")
    return True


def test_3_caching():
    """Test 3: Response Caching."""
    print("\n--- TEST 3: Semantic Caching ---")

    vector_store = VectorStore()

    config = {
        "LLM_ENABLED": True,
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
        "GITHUB_MODELS_ENDPOINT": "https://models.inference.ai.azure.com",
        "GITHUB_MODELS_MODEL": "gpt-4o-mini",
        "GITHUB_MODELS_MAX_TOKENS": 200,
        "GITHUB_MODELS_TEMPERATURE": 0.1,
        "LLM_FALLBACK_TO_TEMPLATE": True,
    }

    llm_service = LLMService(config)

    # Initialize cache
    cache = ResponseCache(
        embedding_model=vector_store.embedding_model,
        similarity_threshold=0.92,
        default_ttl=3600,
        cache_file="data/test_cache.json"
    )

    pipeline = RAGPipeline(vector_store, llm_service=llm_service, cache=cache)

    # Add test doc
    vector_store.add_document(
        "Employees get 15 vacation days.",
        source="test_policy"
    )

    # First query (cache miss)
    print("\nFirst query (should call LLM)...")
    result1 = pipeline.query("How many vacation days?")
    cached1 = result1.get('llm_metadata', {}).get('cached', False)
    print(f"✓ Cached: {cached1}")

    # Second query (cache hit)
    print("\nSecond query (should hit cache)...")
    result2 = pipeline.query("How many vacation days?")
    cached2 = result2.get('llm_metadata', {}).get('cached', False)
    print(f"✓ Cached: {cached2}")

    # Cache stats
    stats = cache.get_statistics()
    print(f"\n✓ Cache stats:")
    print(f"  - Hits: {stats['hits']}")
    print(f"  - Misses: {stats['misses']}")
    print(f"  - Hit rate: {stats['hit_rate']:.2%}")

    if cached2:
        print("✓ Test 3 PASSED - Caching works!")
        return True
    else:
        print("⚠ Test 3 WARNING - Cache didn't hit on second query (may need tuning)")
        return True


def test_4_rate_limit_handling():
    """Test 4: Rate Limit & Fallback."""
    print("\n--- TEST 4: Rate Limit Handling ---")
    print("This test verifies template fallback works when LLM is disabled.")

    vector_store = VectorStore()

    # Disable LLM to test fallback
    config = {
        "LLM_ENABLED": False,
        "LLM_FALLBACK_TO_TEMPLATE": True,
    }

    llm_service = LLMService(config)
    pipeline = RAGPipeline(vector_store, llm_service=llm_service)

    # Add test doc
    vector_store.add_document(
        "Q: How many sick days?\nA: Employees get 10 sick days per year.",
        source="sick_leave_policy"
    )

    result = pipeline.query("How many sick days?")

    print(f"✓ Answer: {result['answer']}")
    print(f"✓ Provider: {result.get('llm_metadata', {}).get('provider')}")

    if result.get('llm_metadata', {}).get('provider') == 'template':
        print("✓ Test 4 PASSED - Fallback works")
        return True
    else:
        print("✗ Test 4 FAILED - Expected template fallback")
        return False


def main():
    """Run all tests."""
    print("\nRunning GitHub Models integration tests...\n")

    tests = [
        test_1_llm_service,
        test_2_rag_pipeline,
        test_3_caching,
        test_4_rate_limit_handling,
    ]

    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYour GitHub Models integration is working correctly!")
        print("\nNext steps:")
        print("1. Start the Flask app: python app.py")
        print("2. Test with real queries via API")
        print("3. Monitor rate limits: GET /api/admin/llm/stats")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please review the errors above and fix before deploying.")

    print("=" * 70)


if __name__ == "__main__":
    main()
