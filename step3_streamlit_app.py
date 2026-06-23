"""
=============================================================================
 STEP 3: STREAMLIT APP — Web UI that uses the LLM Agent + MCP Server
=============================================================================

This is the user-facing web app.
- User types a question in natural language
- Agent (Gemini) decides which MCP tool to call
- Tool result is processed and returned as a natural language answer

Run: streamlit run step3_streamlit_app.py
     (Make sure step1_mcp_server.py is running first!)
"""

import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
import sys
import traceback

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
MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Helper: Convert MCP tools to Gemini format
# ---------------------------------------------------------------------------

def mcp_tools_to_gemini_declarations(mcp_tools) -> list:
    declarations = []
    for tool in mcp_tools:
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
# Core: Agent function (async)
# ---------------------------------------------------------------------------

async def agent_query(question: str) -> dict:
    """
    Run the full agent loop:
      1. Connect to MCP → discover tools
      2. Send question + tools to Gemini
      3. If Gemini calls a tool → execute via MCP → get final answer
      4. Return answer + metadata (which tool was called, etc.)
    """
    result = {"answer": "", "tool_called": None, "tool_args": None, "tool_result": None}

    try:
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Discover tools
                tools_response = await session.list_tools()
                mcp_tools = tools_response.tools

                # Convert to Gemini format
                gemini_declarations = mcp_tools_to_gemini_declarations(mcp_tools)
                gemini_tools = types.Tool(function_declarations=gemini_declarations)

                # Ask Gemini
                client = genai.Client(api_key=GEMINI_API_KEY)

                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Content(role="user", parts=[types.Part(text=question)])
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

                candidate = response.candidates[0]
                part = candidate.content.parts[0]

                if part.function_call:
                    func_name = part.function_call.name
                    func_args = dict(part.function_call.args) if part.function_call.args else {}

                    result["tool_called"] = func_name
                    result["tool_args"] = func_args

                    # Execute tool via MCP
                    tool_response = await session.call_tool(func_name, func_args)
                    tool_text = tool_response.content[0].text
                    result["tool_result"] = tool_text

                    # Send back to Gemini for natural language answer
                    # Use model's actual content (includes thought_signature for gemini-3.1)
                    model_content = candidate.content

                    followup = client.models.generate_content(
                        model=MODEL,
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=question)]),
                            model_content,
                            types.Content(role="user", parts=[
                                types.Part(function_response=types.FunctionResponse(
                                    name=func_name,
                                    response={"result": tool_text}
                                ))
                            ]),
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction="Use the tool result to give a clear, helpful answer.",
                        ),
                    )
                    result["answer"] = followup.candidates[0].content.parts[0].text
                else:
                    result["answer"] = part.text

    except Exception as e:
        error_str = str(e)
        # Check if it's a connection error (TaskGroup wraps connection errors)
        if "ConnectError" in error_str or "connection" in error_str.lower():
            result["answer"] = (
                f"❌ Cannot connect to MCP server at {MCP_SERVER_URL}\n\n"
                f"Please make sure to start the MCP server first:\n"
                f"```\npython step1_mcp_server.py\n```\n\n"
                f"Then try your query again."
            )
        else:
            result["answer"] = f"Error: {error_str}\n\nFull traceback:\n{traceback.format_exc()}"
        
        traceback.print_exc()

    return result


def run_agent(question: str) -> dict:
    """Synchronous wrapper with robust event loop handling."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(agent_query(question))
    except Exception as e:
        error_msg = f"Agent Error: {str(e)}"
        st.error(error_msg)
        traceback.print_exc()
        return {"answer": error_msg, "tool_called": None, "tool_args": None, "tool_result": None}
    finally:
        try:
            loop.close()
        except:
            pass


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Search & Weather Agent", page_icon="🔍", layout="wide")
st.title("🔍 AI Agent — Web Search & Weather (MCP-Powered)")
st.caption("Ask anything — the AI decides whether to search the web, check weather, or answer directly.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_info"):
            with st.expander("🔧 Tool Call Details"):
                st.json(msg["tool_info"])

# Chat input
if prompt := st.chat_input("Ask anything — search the web, check weather..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = run_agent(prompt)

                st.markdown(response["answer"])

                # Show tool details if a tool was called
                tool_info = None
                if response["tool_called"]:
                    tool_info = {
                        "tool": response["tool_called"],
                        "arguments": response["tool_args"],
                        "raw_result": response["tool_result"],
                    }
                    with st.expander("🔧 Tool Call Details"):
                        st.json(tool_info)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "tool_info": tool_info,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

# Sidebar — show available tools
with st.sidebar:
    st.header("Available MCP Tools")
    
    # Status check
    st.markdown("### ⚠️ Prerequisites")
    st.markdown(f"""
    **MCP Server Status:** Make sure to run the MCP server first!
    ```bash
    python step1_mcp_server.py
    ```
    The server should be running at: `{MCP_SERVER_URL}`
    """)
    
    st.divider()
    
    st.markdown("""
    ### Tools Available
    The AI agent dynamically calls these tools:
    
    | Tool | Purpose |
    |------|---------|
    | `web_search` | Search the internet (DuckDuckGo) |
    | `weather_forecast` | Get weather for any city (Open-Meteo) |
    """)

    st.divider()
    st.markdown("**Example questions:**")
    st.markdown("""
    - "What is the latest news about AI?"
    - "Search for Python MCP server tutorial"
    - "What's the weather in Hyderabad?"
    - "Will it rain in London tomorrow?"
    - "What is the capital of France?" (answered directly)
    """)
