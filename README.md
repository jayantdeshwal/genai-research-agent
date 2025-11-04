🚀 GenAI Research Agent

This is a portfolio project demonstrating a "Chat with Search" agent built in Python using LangChain and Streamlit. The agent can hold a conversation and autonomously use tools to find information from the web.

It is deployed live on Streamlit Community Cloud.

✨ Features

Conversational Memory: The agent remembers the last few messages in the conversation.

Agentic Behavior: Built with LangChain's create_agent framework, the agent can reason and decide which tool to use.

Multi-Tool Use: The agent has access to three different tools to answer questions:

DuckDuckGo: For general web searches.

Wikipedia: For factual, encyclopedic knowledge.

Arxiv: For academic papers and research.

Real-time UI: Built with Streamlit, including chat elements and callbacks to show the agent's "thoughts" (if configured).

🛠️ Tech Stack

Python

Streamlit (for the web UI)

LangChain (for the agent framework)

Groq (for the high-speed llama-3.1-8b-instant LLM)

Tools: DuckDuckGo, Wikipedia, Arxiv