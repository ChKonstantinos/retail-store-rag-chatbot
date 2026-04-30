from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src.config import (
    OPENAI_API_KEY,
    LLM_PROVIDER,
    OPENAI_CHAT_MODEL,
)
from src.retriever import get_retrieval_results
from src.prompt_templates import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE


def generate_openai_answer(prompt: str) -> str:
    """
    Generate answer using OpenAI chat model.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found. Please add it to your .env file."
        )

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content


def generate_fallback_answer(query: str, sources: list) -> str:
    """
    Rule-based business-friendly fallback answer.
    Uses retrieved source metadata to produce more specific guidance
    without calling an LLM.
    """

    if not sources:
        return "I could not find this procedure in the available company documents."

    top_source = sources[0]
    title = top_source.get("title", "Unknown Document")

    title_lower = title.lower()
    query_lower = query.lower()

    out_of_scope_terms = [
    "salary", "salaries", "pay raise", "raise",
    "promotion", "bonus", "contract",
    "hiring", "firing", "disciplinary",
    "vacation", "leave", "payroll"
]

    if any(term in query_lower for term in out_of_scope_terms):
        return """
I could not find this procedure in the available company documents.

This question appears to be outside the current Retail Store SOP knowledge base.

Recommended action:
1. Do not provide an unofficial answer.
2. Escalate the question to HR or the store manager.
3. Add the relevant HR policy document to the knowledge base if this topic should be supported.
"""

    if "returns" in title_lower or "refund" in title_lower:
        guidance = """
1. Ask the customer for the original receipt.
2. If there is no receipt, try to locate the transaction using phone number, loyalty card, or payment method.
3. If the transaction cannot be verified, do not issue a cash refund.
4. Store credit may be considered only with store manager approval.
5. Escalate any customer conflict to the store manager.
"""

    elif "cash register" in title_lower:
        guidance = """
1. Follow the cash register procedure for the relevant shift stage.
2. Verify cash, card receipts, vouchers, and POS records.
3. Supervisor approval is required for discounts or manual price changes.
4. At closing, compare the closing balance with the POS report.
5. Report and document any cash difference to the store manager.
"""

    elif "inventory" in title_lower or "receiving" in title_lower:
        guidance = """
1. Compare the delivery note with the physical products received.
2. Check quantities, product codes, expiration dates, and visible damages.
3. Separate damaged items from sellable stock.
4. Report missing items to the supplier or warehouse team.
5. Inventory adjustments must be approved by the store manager.
"""

    else:
        guidance = """
1. Review the relevant company procedure before taking action.
2. Follow only the approved documented process.
3. If the situation is unclear or exceptional, escalate to the store manager.
4. Do not create unofficial exceptions outside the documented policy.
"""

    return f"""
Based on the available company procedure, follow these steps:

{guidance}

Relevant source:
{title}

Note:
This response is generated in retrieval-only mode using rule-based business logic.
For full natural-language answers, activate an LLM provider.
"""


def generate_answer(
    query: str,
    k: int = 3
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    user query -> retrieve context -> generate answer.
    """

    retrieval = get_retrieval_results(query=query, k=k)

    prompt = RAG_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=retrieval["context"],
        question=query
    )

    if LLM_PROVIDER == "openai":
        answer = generate_openai_answer(prompt)

    elif LLM_PROVIDER == "fallback":
        answer = generate_fallback_answer(
            query=query,
            sources=retrieval["sources"]
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
            "Use 'openai' or 'fallback'."
        )

    return {
        "query": query,
        "answer": answer,
        "sources": retrieval["sources"],
        "retrieved_context": retrieval["context"],
        "prompt": prompt
    }