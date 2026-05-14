"""
Galileo Span Types — Sample Application

Demonstrates all five Galileo span types using the @log decorator:
  - Workflow:  the top-level container span
  - Agent:     a reasoning/planning step
  - Retriever: a RAG retrieval step (mocked — no vector DB needed)
  - Tool:      an external tool call (mocked)
  - LLM:       automatic via the Galileo OpenAI wrapper

Run this script and open your Galileo dashboard to see a trace with
the full nested span hierarchy.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 03_sample_app.py
"""

import os
from dotenv import load_dotenv
from galileo.openai import openai
from galileo import log, galileo_context

load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT_NAME")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Mock retriever — behaves like a real RAG retrieval step,
# returns a list of document strings that Galileo captures as retrieved docs.
# ---------------------------------------------------------------------------

@log(span_type="retriever", name="knowledge_base_search")
def retrieve_documents(query: str) -> list[str]:
    """Simulated retrieval — returns static docs as if from a vector store."""
    knowledge_base = {
        "refund": [
            "Refunds are processed within 5–7 business days.",
            "To request a refund, contact support with your order number.",
        ],
        "shipping": [
            "Standard shipping takes 3–5 business days.",
            "Express shipping is available for an additional fee.",
        ],
        "account": [
            "You can reset your password from the login page.",
            "Account settings are available under your profile menu.",
        ],
    }
    # Simple keyword match to simulate semantic search
    for keyword, docs in knowledge_base.items():
        if keyword in query.lower():
            return docs
    return ["No specific documentation found for this query."]


# ---------------------------------------------------------------------------
# Mock tool — simulates an external system call (e.g. a CRM lookup).
# ---------------------------------------------------------------------------

@log(span_type="tool", name="lookup_order_status")
def lookup_order_status(order_id: str) -> dict:
    """Simulated tool: look up an order status from an external system."""
    mock_orders = {
        "ORD-123": {"status": "Shipped", "eta": "2 days"},
        "ORD-456": {"status": "Processing", "eta": "5 days"},
    }
    return mock_orders.get(order_id, {"status": "Not found", "eta": None})


# ---------------------------------------------------------------------------
# Agent reasoning step — decides which tool or retrieval to use.
# ---------------------------------------------------------------------------

@log(span_type="agent", name="plan_response")
def plan_response(user_query: str) -> str:
    """Agent reasoning: decide how to handle the query."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a planning agent. Given a user query, decide the best "
                    "approach: 'retrieve' (search knowledge base), 'tool' (look up order), "
                    "or 'direct' (answer directly). Reply with just one word."
                ),
            },
            {"role": "user", "content": user_query},
        ],
    )
    return response.choices[0].message.content.strip().lower()


# ---------------------------------------------------------------------------
# Final LLM call — generates the answer using retrieved context.
# ---------------------------------------------------------------------------

def generate_answer(user_query: str, context: str) -> str:
    """Generate a final answer using the gathered context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support assistant. Use the provided context to answer the user's question.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}",
            },
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Workflow span — the top-level entry point; becomes the trace root.
# ---------------------------------------------------------------------------

@log(span_type="workflow", name="support_agent_workflow")
def handle_query(user_query: str) -> str:
    """
    Top-level workflow: orchestrates planning, retrieval/tool use, and response.
    This function becomes the root workflow span in Galileo.
    """
    print(f"\nQuery: {user_query}")

    # Step 1: Agent decides what to do
    plan = plan_response(user_query)
    print(f"  Plan: {plan}")

    context = ""

    # Step 2: Execute the plan
    if plan == "retrieve":
        docs = retrieve_documents(user_query)
        context = "\n".join(docs)
        print(f"  Retrieved {len(docs)} document(s)")

    elif plan == "tool":
        # Extract a mock order ID from the query for the demo
        order_id = "ORD-123" if "123" in user_query else "ORD-456"
        result = lookup_order_status(order_id)
        context = f"Order status: {result['status']}, ETA: {result['eta']}"
        print(f"  Tool result: {context}")

    else:
        context = "No additional context needed."

    # Step 3: Generate the final answer (automatic LLM span via OpenAI wrapper)
    answer = generate_answer(user_query, context)
    print(f"  Answer: {answer}\n")
    return answer


# ---------------------------------------------------------------------------
# Run sample queries — each produces a full span hierarchy in Galileo.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with galileo_context(project=project, log_stream=log_stream):
        galileo_context.start_session(name="Lab 03 - Spans")   
        handle_query("How do I get a refund for my order ID ORD-456?")
        handle_query("What kind of shipping options do you offer?")
        handle_query("How can I manage my account?")

    print("✓ Done — open your Galileo dashboard to inspect the span hierarchy.")
    print(f"  Project:    {project}")
    print(f"  Log Stream: {log_stream}")
