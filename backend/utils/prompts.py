"""
Prompt templates for LLM response generation in HR RAG system.

Optimized for:
- Factual accuracy and grounding in source documents
- Professional HR tone
- Source citation
- Concise but complete responses
"""

from typing import List, Dict


class PromptBuilder:
    """Build optimized prompts for HR policy queries."""

    SYSTEM_PROMPT = """You are a professional HR assistant helping employees understand company policies.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. Cite sources using [Source N] notation when referencing information
3. If the context doesn't contain the answer, clearly state this
4. Be concise but thorough
5. Use professional, friendly tone
6. If information is ambiguous or conflicting, mention it

DO NOT:
- Invent or assume information not in the context
- Provide personal opinions or advice beyond stated policies
- Reference external sources or general knowledge"""

    @staticmethod
    def build_rag_prompt(
        question: str,
        contexts: List[Dict],
        enable_citations: bool = True
    ) -> str:
        """
        Build prompt for RAG answer generation.

        Args:
            question: User's question
            contexts: List of retrieved context dicts with 'text', 'source', 'relevance_score'
            enable_citations: Whether to request source citations

        Returns:
            Formatted prompt string optimized for GPT-4o-mini
        """
        # Format contexts with source labels
        context_sections = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("source", "unknown")
            score = ctx.get("relevance_score", 0.0)
            text = ctx.get("text", "").strip()

            context_sections.append(
                f"[Source {i}] (from {source}, relevance: {score:.2f})\n{text}"
            )

        context_text = "\n\n".join(context_sections)

        # Citation instruction
        citation_note = ""
        if enable_citations:
            citation_note = "Cite your sources using [Source N] notation."

        # Build complete prompt
        prompt = f"""{PromptBuilder.SYSTEM_PROMPT}

---
CONTEXT DOCUMENTS:

{context_text}

---
EMPLOYEE QUESTION:
{question}

---
INSTRUCTIONS:
Based solely on the context above, provide a clear and helpful answer. {citation_note}

If the context doesn't contain sufficient information to answer the question, respond with:
"I don't have enough information in the available HR policies to fully answer this question. Please contact HR directly for clarification."

ANSWER:"""

        return prompt

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters).
        For accurate counting, use tiktoken library.
        """
        return len(text) // 4
