import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# --- 1. USE THE MODERN CREATE_AGENT ---
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# --- 2. WE DO NOT NEED LANGCHAIN-HUB OR LANGCHAIN-CLASSIC ---
# from langchain_classic import hub 
from dotenv import load_dotenv

# --- TOOLS ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="GenAI Research Agent",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 GenAI Research Agent")
st.caption("A modern, tool-calling AI agent. (Note: 'Thoughts' are not visible with this agent type).")

# --- POLISHED SIDEBAR ---
st.sidebar.title("Controls & About")
st.sidebar.markdown("This app uses a modern LangChain `create_agent` (Tool-Calling) to answer questions.")
st.sidebar.divider()

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", help="Get your free key from https://console.groq.com/keys")

st.sidebar.divider()
st.sidebar.markdown("### Tech Stack")
st.sidebar.markdown("""
- Streamlit (UI)
- LangChain 1.0+ (Agent Framework)
- Groq (LLM: Llama 3.1 8B)
- DuckDuckGo (Web Search)
- Wikipedia (Knowledge Base)
- Arxiv (Research Papers)
""")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a research agent. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("What is the latest on Llama 3.1?"):
    
    # --- API KEY CHECK ---
    if not api_key:
        st.info("Please add your Groq API Key in the sidebar to continue.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant", 
        streaming=True,
    )

    tools = [search, arxiv, wiki]

    # --- 3. THIS IS THE MODERN PROMPT (A simple string!) ---
    # We are NOT using the old 'react-chat' prompt.
    # This just gives the agent a role. The "how-to-think"
    # is already built into the 'create_agent' function.
    system_prompt_string = "You are a helpful research assistant. Use your tools to answer the user's question."


    search_agent = create_agent(
        model=llm,
        tools=tools,
        # --- 4. PASS THE SIMPLE PROMPT HERE ---
        system_prompt=system_prompt_string
    )

    with st.chat_message("assistant"):
        # --- 5. CALLBACK HANDLER (Thoughts will NOT expand) ---
        st_cb = StreamlitCallbackHandler(
            st.container(), 
            expand_new_thoughts=False # We set to False, as this agent has no "thoughts"
        )
        
        # --- 6. ADD ERROR HANDLING ---
        try:
            # --- 7. USE THE CORRECT INPUT FORMAT FOR CREATE_AGENT ---
            response = search_agent.invoke(
                {"messages": st.session_state.messages}, 
                callbacks=[st_cb]
            )
            # --- 8. GET THE OUTPUT FROM THE CORRECT KEY ---
            content = response["messages"][-1].content

        except Exception as e:
            # --- This runs if the "try" block fails ---
            st.error("Sorry, something went wrong. Please check your API key or try again.")
            print(e)  # This will print the error to your Streamlit logs
            content = None # Don't save a bad response
            
        if content:
            st.session_state.messages.append({"role": "assistant", "content": content})
            st.write(content)

