import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic import hub # <-- Standardized this import
from dotenv import load_dotenv

# --- PAGE CONFIG (NEW) ---
# Set the page configuration. This MUST be the first Streamlit command.
st.set_page_config(
    page_title="Chat with Search", # <-- NEW: Title in browser tab
    page_icon="🔎",             # <-- NEW: Icon in browser tab
    layout="centered",          # <-- NEW: Makes chat look cleaner
    initial_sidebar_state="expanded"
)

# --- LOAD ENV (Added) ---
load_dotenv()

# --- TOOLS ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --- STREAMLIT UI ---
st.title("🔎 LangChain - Chat with Search")
# (NEW) Add a caption for polish
st.caption("I'm a smart agent that can search the web, Wikipedia, and Arxiv for you.")

# (CHANGED) Sidebar polish
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.divider() # (NEW)
    st.markdown("Powered by LangChain, Groq, & Streamlit") # (NEW)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a GenAI agent. How can I help you with your research today?"}
    ]

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# (CHANGED) Add placeholder
if prompt := st.chat_input(placeholder="What is the future of Generative AI?"):
    
    # --- API Key Check (NEW) ---
    if not api_key:
        st.info("Please enter your Groq API Key in the sidebar to start.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --- AGENT LOGIC ---
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant", 
        streaming=True,
    )
    
    tools = [search, arxiv, wiki]

    # --- THE FIX ---
    prompt_template = hub.pull("hwchase17/react-chat")

    search_agent = create_agent(
        model=llm,
        tools=tools,
        # (CHANGED) Pass the full prompt object, not just the string
        system_prompt=prompt_template.template
    )

    with st.chat_message("assistant"):
        # (CHANGED) This is the "attractive" part - show the thoughts!
        st_cb = StreamlitCallbackHandler(
            st.container(), 
            expand_new_thoughts=True, # <-- CHANGED
            collapse_completed_thoughts=False # <-- NEW
        )
        response = search_agent.invoke({"messages": st.session_state.messages}, callbacks=[st_cb])

        # Extract and display the final text response
        content = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.write(content)


