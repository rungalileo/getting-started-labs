"""
Graph View Demo — Galileo Observability
=======================================
Demonstrates how to instrument a multi-step research agent with Galileo
so that every function appears as a named node in Graph View.

The agent:
  1. Retrieves mock context for a given question
  2. Synthesizes an answer using the OpenAI API (wrapped by Galileo)
  3. Validates the answer against a simple quality check

All LLM calls are automatically logged via `galileo.openai`.
Each function decorated with `@log` becomes a named span (node) in the trace graph.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 04_sample_app.py
"""

import os
import sys
from dotenv import load_dotenv
from galileo.openai import openai
from galileo import log, galileo_context

load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")

#client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
# ── Sample data: mock retrieval corpus ────────────────────────────────────────

KNOWLEDGE_BASE = {
    "climate": "Global temperatures have risen ~1.2°C above pre-industrial levels. "
               "The IPCC projects further warming of 1.5–4.5°C by 2100 depending on emissions.",
    "ai":      "Large language models are trained on vast text corpora using self-supervised learning. "
               "They exhibit emergent capabilities at scale, including reasoning and code generation.",
    "space":   "NASA's Artemis program aims to return humans to the Moon by 2026. "
               "SpaceX's Starship is central to lunar landing missions.",
    "default": "I have general knowledge on many topics but couldn't find a specific match."
}


# ── Agent steps — each decorated with @log becomes a node in Galileo Graph View ─

@log(span_type="retriever", name="retrieve_context")
def retrieve_context(question: str) -> str:
    """Retrieve relevant context from the mock knowledge base."""
    question_lower = question.lower()
    for keyword, context in KNOWLEDGE_BASE.items():
        if keyword.lower() in question_lower:
            return context
    return KNOWLEDGE_BASE["default"]


@log(span_type="tool", name="synthesize_answer")
def synthesize_answer(question: str, context: str) -> str:
    """Call the LLM to generate an answer grounded in the retrieved context."""
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise and accurate. If the context is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()


@log(span_type="tool", name="validate_answer")
def validate_answer(answer: str) -> dict:
    """Run a simple quality check on the generated answer."""
    word_count = len(answer.split())
    has_content = word_count > 10
    is_too_long = word_count > 200

    return {
        "passed": has_content and not is_too_long,
        "word_count": word_count,
        "issues": (
            []
            if has_content and not is_too_long
            else (
                ["answer_too_short"] if not has_content else ["answer_too_long"]
            )
        ),
    }


@log(span_type="agent", name="run_agent")
def run_agent(question: str) -> dict:
    """
    Top-level agent entry point.
    This function ties together retrieval → synthesis → validation.
    In Graph View, you'll see this as the root span with three child nodes.
    """
    print(f"\nQuestion: {question}")

    context = retrieve_context(question)
    print(f"  [retrieve] matched context ({len(context)} chars)")

    answer = synthesize_answer(question, context)
    print(f"  [synthesize] generated answer ({len(answer.split())} words)")

    validation = validate_answer(answer)
    status = "✓ passed" if validation["passed"] else f"✗ failed: {validation['issues']}"
    print(f"  [validate] {status}")

    return {
        "question": question,
        "answer": answer,
        "validation": validation,
    }


# ── Main: run several questions to generate diverse traces ────────────────────

if __name__ == "__main__":
    with galileo_context(project=project, log_stream=log_stream):
        galileo_context.start_session(name="Lab 04 - Agent Graph")
        questions = [
            "What is happening with climate change?",
            "How do AI large language models work?",
            "What are NASA's plans for space exploration?",
            "Tell me about the history of jazz music.",  # will hit the "default" branch
        ]

        results = []
        for q in questions:
            result = run_agent(q)
            results.append(result)

        print("\n" + "=" * 60)
        print(f"Completed {len(results)} agent runs.")
    print("Open app.galileo.ai → your project → Traces to see results.")
    print("Switch to Graph tab on any trace to see the node graph.")
    print("Navigate to Agent Graph in the sidebar for the aggregate view.")
