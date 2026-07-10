"""
Agent Observability with Galileo — Sample Application
Topic: Log Streams

This script demonstrates a simple tool-using agent instrumented with Galileo.
Galileo captures every LLM call, tool invocation, and span so you can see
exactly what the agent did, why, and where things went wrong.

This script assumes the .env file is located
in the parent directory, and it's properly configured.

Setup:
    1. review the .env configuration file
    2. pip install -r requirements.txt
    3. python 02_sample_app.py
"""

import os
import json
from dotenv import load_dotenv
from galileo import log, galileo_context
from galileo.openai import openai


load_dotenv("../.env")
project=os.environ.get("GALILEO_PROJECT")
log_stream=os.environ.get("GALILEO_LOG_STREAM")

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


# Confirm that environment variables are properly mapped
print(f"Loaded env variables: galileo console: {os.environ.get("GALILEO_CONSOLE_URL")} | Project: {project} | Log Stream: {log_stream}")


# --- Simple tools the agent can call ---

@log(span_type="tool")
def get_account_balance(account_id: str) -> dict:
    """Simulated tool: fetch account balance."""
    # Simulated data — in a real app this would call your backend
    balances = {
        "ACC-001": {"balance": 4200.00, "currency": "USD"},
        "ACC-002": {"balance": 150.75, "currency": "USD"},
    }
    return balances.get(account_id, {"error": f"Account {account_id} not found"})

@log(span_type="tool")
def check_loan_eligibility(account_id: str, loan_amount: float) -> dict:
    """Simulated tool: check if an account is eligible for a loan."""
    balance = get_account_balance(account_id).get("balance", 0)
    eligible = balance >= loan_amount * 0.2  # Simple rule: need 20% of loan as balance
    return {
        "eligible": eligible,
        "account_id": account_id,
        "requested_amount": loan_amount,
        "reason": "Sufficient balance" if eligible else "Insufficient balance for loan",
    }


# --- Tool definitions for the LLM ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the current balance for a bank account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID, e.g. ACC-001",
                    }
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_loan_eligibility",
            "description": "Check if an account is eligible for a loan of a given amount",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID",
                    },
                    "loan_amount": {
                        "type": "number",
                        "description": "The requested loan amount in USD",
                    },
                },
                "required": ["account_id", "loan_amount"],
            },
        },
    },
]

TOOL_MAP = {
    "get_account_balance": get_account_balance,
    "check_loan_eligibility": check_loan_eligibility,
}


# --- Agent loop instrumented with Galileo ---

@log(span_type="agent", name="Agent Workflow")
def run_agent(user_query: str):
    """
    Run a simple tool-using agent and log every step to Galileo.
    This demonstrates agent observability: every LLM call, tool call,
    and final response is captured as a trace with spans.
    """


    print(f"\n{'='*60}")
    print(f"User query: {user_query}")
    print(f"{'='*60}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful banking assistant. "
                "Use the available tools to answer questions about accounts and loans. "
                "Always use the exact account ID provided by the user."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    # Agentic loop — runs until the model stops calling tools
    max_steps = 5
    for step in range(max_steps):
        print(f"\n[Step {step + 1}] Calling LLM...")

        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_NAME"),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message


        # If no tool calls, we're done
        if not message.tool_calls:
            print(f"\nFinal answer: {message.content}")
            break

        # Otherwise, process each tool call
        messages.append(message)
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  -> Tool call: {fn_name}({fn_args})")

            # Execute the tool
            result = TOOL_MAP[fn_name](**fn_args)
            print(f"  <- Tool result: {result}")


            # Append tool result to message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })



# --- Run sample queries ---

if __name__ == "__main__":
    with galileo_context(project=project, log_stream=log_stream):
        galileo_context.start_session(name="Lab 02 - Log Streams")

        # Query 1: Normal flow — should work fine
        run_agent("What is the balance for account ACC-001?")

        # Query 2: Loan eligibility check
        run_agent("Can account ACC-002 get a loan of $5000?")

        # Query 3: Ambiguous query — watch how the agent handles it
        run_agent("Can I get a loan?")

    print("\n\nOpen your Galileo dashboard to see the traces for these 3 runs.")
    print("Look for differences in tool selection quality and context adherence across the queries.")
