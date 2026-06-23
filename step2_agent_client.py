"""
=============================================================================
 STEP 2: LLM AGENT CLIENT — Connects to MCP, discovers tools, calls them
                             dynamically based on user's natural language query
=============================================================================

This is the KEY piece that makes tool calling DYNAMIC.
The LLM (Gemini) reads the tool descriptions from MCP and decides which to call.

Flow:
  1. Connect to MCP server → get list of tools + their descriptions
  2. User asks a question in natural language
  3. Send question + tool definitions to Gemini
  4. Gemini responds with a tool call (or direct answer)
  5. Execute the tool call via MCP
  6. Send tool result back to Gemini for final answer
  7. Return final answer to user

Run: python step2_agent_client.py
     (Make sure step1_mcp_server.py is running first!)
"""

import asyncio
import json
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_SERVER_URL = "http://localhost:8080/sse"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "gemini-3.1-flash-lite"  # Supports function calling


# ---------------------------------------------------------------------------
# Convert MCP tools → Gemini function declarations
# ---------------------------------------------------------------------------

def mcp_tools_to_gemini_declarations(mcp_tools) -> list:
    """
    Convert MCP tool definitions into Gemini-compatible function declarations.
    
    This is the bridge: MCP describes tools in its format,
    Gemini needs them in its function-calling format.
    """
    declarations = []

    for tool in mcp_tools:
        # Build parameter schema from MCP tool's input_schema
        properties = {}
        required = []

        if tool.inputSchema and "properties" in tool.inputSchema:
            for param_name, param_info in tool.inputSchema["properties"].items():
                properties[param_name] = {
                    "type": param_info.get("type", "string").upper(),
                    "description": param_info.get("description", ""),
                }

            required = tool.inputSchema.get("required", [])

        declaration = types.FunctionDeclaration(
            name=tool.name,
            description=tool.description or f"Tool: {tool.name}",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    k: types.Schema(type=v["type"], description=v["description"])
                    for k, v in properties.items()
                },
                required=required,
            ) if properties else None,
        )
        declarations.append(declaration)

    return declarations


# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------

async def ask_agent(question: str) -> str:
    """
    Send a user question to the LLM agent.
    The agent dynamically decides which MCP tools to call (if any).
    
    Returns the final answer.
    """

    # --- 1. Connect to MCP server and discover tools ---
    async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Get all available tools from MCP server
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools

            print(f"\n[Agent] Discovered {len(mcp_tools)} tools from MCP server:")
            for t in mcp_tools:
                print(f"  - {t.name}: {t.description[:50]}...")

            # --- 2. Convert MCP tools to Gemini function declarations ---
            gemini_declarations = mcp_tools_to_gemini_declarations(mcp_tools)
            gemini_tools = types.Tool(function_declarations=gemini_declarations)

            # --- 3. Send user question + tools to Gemini ---
            client = genai.Client(api_key=GEMINI_API_KEY)

            print(f"\n[Agent] Sending to Gemini: '{question}'")
            print(f"[Agent] Gemini will decide which tool(s) to call...\n")

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=question)]
                    )
                ],
                config=types.GenerateContentConfig(
                    tools=[gemini_tools],
                    system_instruction=(
                        "You are a helpful AI assistant with access to web search and weather tools. "
                        "Use web_search when the user asks about current events, facts, or needs information from the internet. "
                        "Use weather_forecast when the user asks about weather for any city. "
                        "If you can answer directly without tools, do so."
                    ),
                ),
            )

            # --- 4. Check if Gemini wants to call a tool ---
            candidate = response.candidates[0]
            part = candidate.content.parts[0]

            # If Gemini returns a function call → execute it via MCP
            if part.function_call:
                func_name = part.function_call.name
                func_args = dict(part.function_call.args) if part.function_call.args else {}

                print(f"[Agent] Gemini decided to call: {func_name}({func_args})")

                # --- 5. Execute the tool via MCP ---
                result = await session.call_tool(func_name, func_args)
                tool_result = result.content[0].text

                print(f"[Agent] Tool returned:\n{tool_result}\n")

                # --- 6. Send tool result back to Gemini for final answer ---
                # Use the model's actual response content (includes thought_signature for gemini-3.1)
                model_content = candidate.content

                followup_response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Content(role="user", parts=[types.Part(text=question)]),
                        model_content,
                        types.Content(role="user", parts=[
                            types.Part(function_response=types.FunctionResponse(
                                name=func_name,
                                response={"result": tool_result}
                            ))
                        ]),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a helpful AI assistant. "
                            "Use the tool result to provide a clear, helpful answer."
                        ),
                    ),
                )

                final_answer = followup_response.candidates[0].content.parts[0].text
                return final_answer

            else:
                # Gemini answered directly without calling a tool
                return part.text


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("  MCP + LLM Agent — Dynamic Tool Calling Demo")
    print("=" * 60)
    print("\nType your questions. The LLM will decide which tools to use.")
    print("Type 'quit' to exit.\n")

    # Demo questions to show dynamic behavior
    demo_questions = [
        "Search the web for 'what is MCP protocol in AI'",
        "What's the weather in Hyderabad?",
        "Weather forecast for London",
        "Search for latest news about artificial intelligence",
        "What is the capital of Japan?",  # Gemini answers directly (no tool needed)
    ]

    print("--- DEMO MODE (5 example questions) ---\n")
    for q in demo_questions:
        print(f"{'─' * 60}")
        print(f"USER: {q}")
        answer = await ask_agent(q)
        print(f"\nAGENT: {answer}\n")

    # Interactive mode
    print("\n--- INTERACTIVE MODE ---\n")
    while True:
        question = input("YOU: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = await ask_agent(question)
        print(f"\nAGENT: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
