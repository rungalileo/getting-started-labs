"""
Production Monitoring Demo — Galileo
======================================
Demonstrates a production-instrumented chatbot application with:
  - Session management (one session per conversation)
  - Full trace and span hierarchy via the @log decorator + Galileo OpenAI wrapper
  - Rich tags and metadata for failure mode analysis
  - Log stream metrics configuration via SDK

Run this script to simulate 5 chatbot conversations being logged to the
'production' Log stream. Then open the Galileo Console to explore:
  - Session-level metric scores
  - Individual trace span trees
  - Filtering by metadata (user tier, prompt version, session type)
  - The Configure Metrics panel to enable and sample metrics

After running, try:
  1. Filter sessions by metric score to find underperformers
  2. Drill into a failing trace and examine each span
  3. Apply a metadata filter: user_tier = "free"
  4. Add a low-scoring trace to your evaluation dataset


This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 08_sample_app.py
"""

import os
import uuid
import time
from dotenv import load_dotenv
from galileo import GalileoMetrics
from galileo.log_streams import enable_metrics
from galileo import GalileoLogger, galileo_context, log
from openai import OpenAI

load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT_NAME")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")

# ---------------------------------------------------------------------------
# Step 1: Enable metrics on the Log stream (runs once at startup)
# In production, call this during app initialization or a setup script.
# ---------------------------------------------------------------------------
try:
    enable_metrics(
        project_name=project,
        log_stream_name=log_stream,
        metrics=[
            GalileoMetrics.instruction_adherence,
            GalileoMetrics.completeness,
        ],
    )
    print("Metrics enabled.\n")
except Exception as e:
    print(f"Note: Could not configure metrics automatically ({e}). Configure in the Console.\n")

# ---------------------------------------------------------------------------
# Initialize the Galileo context for this environment
# ---------------------------------------------------------------------------
galileo_context.init(project=project, log_stream=log_stream)

# ---------------------------------------------------------------------------
# LLM client — Galileo wrapper automatically traces all calls
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

logger = GalileoLogger(project=project, log_stream=log_stream)


SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for a SaaS analytics platform. "
    "Answer questions clearly and concisely. "
    "If you cannot resolve the issue, guide the user to contact support@example.com."
)

# Prompt version tag — update this when you change the system prompt
PROMPT_VERSION = "v2.0"


# ---------------------------------------------------------------------------
# Simulated conversations with rich metadata for failure mode analysis
# ---------------------------------------------------------------------------
conversations = [
    {
        "user_id": "usr-001",
        "user_tier": "enterprise",
        "session_type": "technical-support",
        "turns": [
            "How do I connect my data warehouse to the dashboard?",
            "Does it support BigQuery?",
        ],
    },
    {
        "user_id": "usr-002",
        "user_tier": "free",
        "session_type": "billing",
        "turns": [
            "Why was I charged twice this month?",
            "How do I get a refund?",
        ],
    },
    {
        "user_id": "usr-003",
        "user_tier": "pro",
        "session_type": "feature-request",
        "turns": [
            "Can I schedule automated reports?",
            "Is there a Slack integration?",
        ],
    },
    {
        "user_id": "usr-004",
        "user_tier": "free",
        "session_type": "technical-support",
        "turns": [
            "My dashboard is not loading any data.",
            "I already cleared the cache, still broken.",
            "This is urgent, nothing is working.",  # Frustrated user
        ],
    },
    {
        "user_id": "usr-005",
        "user_tier": "enterprise",
        "session_type": "onboarding",
        "turns": [
            "We just signed up — how do we get started?",
            "Do you offer onboarding calls?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Simulate production traffic — one session per conversation
# ---------------------------------------------------------------------------
for conv in conversations:
    session_name = f"conv-{conv['user_id']}-{conv['session_type']}"
    session_id = logger.start_session(
        name=session_name,
        external_id=str(uuid.uuid4()),  # tie to your internal conversation ID
    )
    print(f"Starting session: {session_name} (tier: {conv['user_tier']})")

    history = []

    for i, user_message in enumerate(conv["turns"]):
        # Attach metadata to every trace for filtering in the Console
        trace_metadata = {
            "user_id": conv["user_id"],
            "user_tier": conv["user_tier"],
            "session_type": conv["session_type"],
            "prompt_version": PROMPT_VERSION,
            "turn_number": str(i + 1),
            "environment": log_stream,
        }
        tags = [conv["user_tier"], conv["session_type"], PROMPT_VERSION]

        logger.start_trace(
            input=user_message,
            name=f"turn-{i + 1}",
            tags=tags,
            metadata=trace_metadata,
        )

        start_ns = time.perf_counter_ns()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        llm_response = client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.3,
            messages=messages,
            max_completion_tokens=300
        )
        response = llm_response.choices[0].message.content
        input_tokens = llm_response.usage.prompt_tokens
        output_tokens = llm_response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens

        end_ns = time.perf_counter_ns()
        elapsed_ns = end_ns - start_ns

        logger.add_llm_span(
            input=messages,
            output=response,
            model="gpt-5.2",
            name="Response Generation",
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ns=elapsed_ns
        )
        
        logger.conclude()
        logger.flush()

        # Update conversation history for multi-turn context
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})

        print(f"  Turn {i + 1}: '{user_message[:60]}...' ✓")

    logger.clear_session()
    print(f"  Session complete.\n")

print("All conversations logged to Galileo.")
print(f"\nOpen the Galileo Console → Project '{project}' → Log Stream '{log_stream}'")
print("\nSuggested next steps:")
print("  1. Filter sessions by metric score (look for instruction_adherence < 0.6)")
print("  2. Drill into a failing trace — examine the span tree")
print("  3. Filter by metadata: user_tier = 'free' to segment failures")
print("  4. Click 'Configure Metrics' to adjust sampling or add new metrics")
print("  5. Export a low-scoring trace to your evaluation dataset")
