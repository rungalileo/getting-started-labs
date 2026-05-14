"""
Out-of-the-Box Metrics Demo — Galileo
======================================
Demonstrates how to attach Galileo's built-in preset metrics
to experiments, which are offline evaluations against a dataset. 

Use case: a simple RAG-style Q&A agent over a fixed knowledge base.

Metrics demonstrated:
  - Correctness            — Is the answer factually right?
  - Completeness           — Is the answer complete?
  - Context Adherence      — Does the answer stay grounded in retrieved context?
  - Instruction Adherence  — Does the answer follow the system prompt rules?
  - Chunk Relevance        — Is the chunk relevant to the question?
  - Chunk Attribution      - Was the chunk used by the LLM to generate the response? 

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 05_sample_app.py
"""

import os
from dotenv import load_dotenv
from galileo.openai import openai
from galileo import log
from galileo.schema.metrics import GalileoMetrics
from galileo.experiments import run_experiment

load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT_NAME")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Mock knowledge base (simulates a vector store retrieval) ──────────────────

KNOWLEDGE_BASE = {
    "climate": (
        "Global average temperatures have risen approximately 1.2°C above pre-industrial levels. "
        "The IPCC's Sixth Assessment Report projects continued warming of 1.5–4.5°C by 2100, "
        "depending on emissions trajectories. The last decade (2011–2020) was the warmest on record."
    ),
    "ai": (
        "Large language models (LLMs) are trained on vast text corpora using self-supervised learning "
        "with transformer architectures. Emergent capabilities such as few-shot reasoning and code "
        "generation appear at scale. RLHF (Reinforcement Learning from Human Feedback) is commonly "
        "used to align LLMs with human preferences."
    ),
    "NASA": (
        "NASA's Artemis program is designed to return humans to the lunar surface. Artemis III, "
        "targeting the lunar south pole, uses SpaceX's Starship as the human landing system. "
        "The program aims to establish a sustained presence on and around the Moon by the late 2020s."
    ),
}

SYSTEM_PROMPT = (
    "You are a concise research assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer under 3 sentences. "
)

# ── Agent functions ────────────────────────────────────────────────────────────

@log(span_type="retriever")
def retrieve_context(question: str) -> str:
    """Retrieve the most relevant context chunk for the question."""
    question_lower = question.lower()
    for keyword, context in KNOWLEDGE_BASE.items():
        if keyword.lower() in question_lower:
            return context
    return ""


@log(span_type="llm")
def generate_answer(question: str, context: str) -> str:
    """Generate an answer grounded in the retrieved context."""
    user_message = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context
        else f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=200,
    )
    return response.choices[0].message.content.strip()


@log(span_type="agent")
def run_rag_agent(question: str) -> dict:
    """Full RAG pipeline: retrieve context then generate a grounded answer."""
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    return {"question": question, "context": context, "answer": answer}


# ── Part 1: Experiment with preset metrics ─────────────────────────────────────

def _run_experiment():
    """
    Run an offline experiment against a fixed dataset, scoring each row
    with Galileo's out-of-the-box metrics.
    Scores appear in the Experiments tab in the Galileo console.
    """

    # Dataset: list of (question, expected_answer) pairs
    dataset = [
        {
            "input": "What is happening with climate change?",
            "expected_output": "Global temperatures have risen ~1.2°C and continue to rise.",
        },
        {
            "input": "How do AI large language models work?",
            "expected_output": "LLMs use transformer architectures trained on large text corpora.",
        },
        {
            "input": "What is NASA's Artemis program?",
            "expected_output": "Artemis aims to return humans to the Moon using SpaceX Starship.",
        },
        {
            "input": "Tell me about the history of jazz music.",  # out-of-domain — no context
            "expected_output": "Jazz originated in New Orleans in the early 20th century.",
        },
    ]


    # Select which metrics to compute
    metrics = [
        GalileoMetrics.correctness,
        GalileoMetrics.completeness,
        GalileoMetrics.context_adherence,
        GalileoMetrics.instruction_adherence,
        GalileoMetrics.chunk_attribution_utilization,
        GalileoMetrics.chunk_relevance
    ]

    print("\n[Experiment] Running offline evaluation with preset metrics...")
    experiment = run_experiment(
        experiment_name="metrics-demo-experiment",
        dataset=dataset,
        function=run_rag_agent,
        metrics=metrics,
        project=project
    )

    print("[Experiment] Done — view results in app.galileo.ai → Experiments")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Galileo Out-of-the-Box Metrics Demo")
    print("=" * 60)

    # Offline experiment with explicit metric selection
    _run_experiment()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Open the Galileo UI → Experiments and find this experiment.")
    print("  2. Navigate throught the traces to see the metrics scores.")