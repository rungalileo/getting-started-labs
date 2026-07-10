"""
Experiments Demo — Galileo
===========================
Demonstrates two approaches to running experiments in Galileo:

  1. Prompt template experiment — Galileo owns the LLM call and variable substitution
  2. Custom function experiment — your application code runs; all spans are traced

Also includes a standalone unit test (test_experiments.py pattern) showing how
to embed experiments in a CI/CD pipeline with a metric threshold assertion.

Run this script to see both experiment approaches in action, then compare the
two results in the Galileo Console using the "Compare Experiments" feature.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 07_sample_app.py
"""

import os
from dotenv import load_dotenv
from galileo import GalileoMetrics, Message, MessageRole, log
from galileo.datasets import create_dataset, get_dataset
from galileo.experiments import run_experiment
from galileo.openai import openai
from galileo.prompts import create_prompt, get_prompt

load_dotenv("../.env")

project=os.environ.get("GALILEO_PROJECT")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")


# ---------------------------------------------------------------------------
# Shared dataset — used by both experiments
# ---------------------------------------------------------------------------
support_data = [
    {
        "input": "How do I reset my password?",
        "ground_truth": "Click 'Forgot Password' on the login page and follow the email link to reset your password.",
    },
    {
        "input": "Can I export my data?",
        "ground_truth": "Yes. Go to Settings > Data Management > Export and choose your preferred format.",
    },
    {
        "input": "How do I invite a teammate?",
        "ground_truth": "Navigate to Settings > Team > Invite Member, enter their email, and select a role.",
    },
    {
        "input": "What happens to my data if I cancel?",
        "ground_truth": "Your data is retained for 30 days after cancellation. You can export it anytime in that window.",
    },
    {
        "input": "Is there a mobile app?",
        "ground_truth": "Yes, we have iOS and Android apps available in the App Store and Google Play respectively.",
    },
]

try:
    print("Creating shared dataset...")
    dataset = create_dataset(name="support-demo-dataset", content=support_data)
    print(f"Dataset created: {len(support_data)} rows.\n")
except Exception as e:
    print("Dataset already exists. Loading it")
    dataset = get_dataset(name="support-demo-dataset")


# ---------------------------------------------------------------------------
# Experiment 1: Prompt template approach
# Galileo handles variable substitution, the LLM call, and metric scoring.
# Use this when you're iterating on the prompt itself.
# ---------------------------------------------------------------------------
print("--- Experiment 1: Prompt Template ---")

try:
    prompt = create_prompt(
        name="support-bot-prompt",
        project_name=project,
        template=[
            Message(
                role=MessageRole.system,
                content=(
                    "You are a concise, helpful customer support assistant. "
                    "Answer the user's question clearly in 1–3 sentences. "
                    "If you don't know the answer, guide them to contact support@example.com."
                ),
            ),
            Message(role=MessageRole.user, content="{{ input }}"),
        ]
    )
except Exception as e:
    print("Prompt already exists. Loading it")
    prompt = get_prompt(name="support-bot-prompt")


results_v1 = run_experiment(
    "support-bot-prompt-v1",
    dataset=dataset,
    prompt_template=prompt,
    metrics=[
        GalileoMetrics.correctness,
        GalileoMetrics.completeness,
    ],
    project=project,
)

print("Experiment 1 complete. View at the link above.\n")

# ---------------------------------------------------------------------------
# Experiment 2: Custom function approach
# Your application code runs; the @log decorator captures all spans.
# Use this when you want to test real application logic — not just the prompt.
# ---------------------------------------------------------------------------
print("--- Experiment 2: Custom Function ---")

#client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


SYSTEM_PROMPT = (
    "You are a concise, helpful customer support assistant. "
    "Answer the user's question clearly in 1–3 sentences. "
    "If you don't know the answer, guide them to contact support@example.com."
)


@log
def support_bot(input: dict) -> str:
    """
    The actual support bot function.
    Using @log means every LLM call inside is traced as a span.
    Galileo's experiment runner creates a new trace per dataset row —
    do NOT start a new trace or flush manually here.
    """
    query = input["input"] if isinstance(input, dict) else input
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_completion_tokens=300
    )
    return response.choices[0].message.content


results_v2 = run_experiment(
    "support-bot-function-v1",
    dataset=dataset,
    function=support_bot,
    metrics=[
        GalileoMetrics.correctness,
        GalileoMetrics.completeness,
    ],
    project=project,
)

print("Experiment 2 complete. View at the link above.\n")

# ---------------------------------------------------------------------------
# Summary: how to compare in the Console
# ---------------------------------------------------------------------------
print("Both experiments are logged. To compare:")
print("  1. Open the Experiments tab in the Galileo Console")
print("  2. Check the boxes next to 'support-bot-prompt-v1' and 'support-bot-function-v1'")
print("  3. Click 'Compare Experiments'")
print("  4. Review metrics, outputs, latency, and token usage side by side.\n")

# ---------------------------------------------------------------------------
# CI/CD pattern: embed in a unit test
# Run with: python test_support_bot.py
# ---------------------------------------------------------------------------
