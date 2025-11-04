import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic import hub
from dotenv import load_dotenv

# --- TOOLS ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --- STREAMLIT UI ---
st.title("🔎 LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant", 
        streaming=True,
    )

    tools = [search, arxiv, wiki]

    # --- THE FIX ---
    # 2. Pull the standard "ReAct" prompt (the "brain" of the agent)
    prompt_template = hub.pull("hwchase17/react-chat")

    search_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt_template.template
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
                st.container(), 
                expand_new_thoughts=True,     # <-- Set to True to see thoughts by default
                collapse_completed_thoughts=False, # <-- Keep thoughts visible
                max_thought_containers=6,     # <-- Limit number of thoughts shown
            )
        response = search_agent.invoke({"messages": st.session_state.messages}, callbacks=[st_cb])

       # Extract and display the final text response
        content = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.write(content)
