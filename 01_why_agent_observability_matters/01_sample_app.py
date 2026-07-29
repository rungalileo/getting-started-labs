"""
Log Streams with Galileo — Sample Application

Demonstrates how to send traces to a Galileo Log Stream using the
Galileo OpenAI wrapper. All LLM calls are automatically logged —
no manual instrumentation needed.

This script runs a few different prompts; open your Galileo dashboard
to see the traces appear in your Log Stream.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 01_sample_app.py
"""

import os
from dotenv import load_dotenv
from galileo.openai import openai
from galileo import galileo_context

load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT_NAME")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")

# Galileo wraps the OpenAI client automatically.
# Traces are sent to the project and log stream set in your .env file.
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def ask(question: str) -> str:
    """Send a question to the model and return the response."""
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    return answer


if __name__ == "__main__":
    # Run a few sample prompts — each becomes a trace in your Log Stream
    with galileo_context(project=project, log_stream=log_stream):
        galileo_context.start_session(name="Lab 01 - Why Agent Observability Matters")
        ask("Summarize the impact of rising interest rates on technology sector and how it might affect valuations and investor sentiment in the short term.")
        ask("What are the best growth stocks to invest in?")
        ask("What would be the impact of rising interest rages?")

    print("✓ Done — open your Galileo dashboard to see the traces.")
    print(f"  Project:    {os.environ.get('GALILEO_PROJECT_NAME')}")
    print(f"  Log Stream: {os.environ.get('GALILEO_LOG_STREAM')}")
