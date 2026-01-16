import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise GenAI RAG", layout="wide")
st.title("📄 Enterprise GenAI Research Assistant")

# ---------------------------
# Sidebar: Upload PDF + Retriever Selector
# ---------------------------
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

st.sidebar.header("🔍 Retriever Settings")
retriever_type = st.sidebar.selectbox(
    "Choose Retriever",
    ["faiss", "bm25", "hybrid"],
    index=0
)

if uploaded_file:
    with st.spinner("Indexing document..."):
        files = {"file": uploaded_file}
        res = requests.post(f"{API_URL}/upload", files=files)

    if res.status_code == 200:
        st.sidebar.success("✅ Document indexed successfully!")
    else:
        st.sidebar.error("❌ Upload failed")

# ---------------------------
# Chat Section
# ---------------------------
st.subheader("💬 Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask a question about the document...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.get(
                    f"{API_URL}/ask",
                    params={
                        "query": query,
                        "retriever_type": retriever_type
                    },
                    timeout=60
                )

                if res.status_code != 200:
                    st.error(f"❌ API Error: {res.status_code}")
                    st.stop()

                data = res.json()

                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                used_retriever = data.get("retriever", retriever_type)

                st.markdown(f"### ✅ Answer ({used_retriever})")
                st.markdown(answer)

                with st.expander("📚 Sources"):
                    if not sources:
                        st.write("No sources returned.")
                    else:
                        for i, s in enumerate(sources, start=1):
                            page = s.get("page", "?")
                            text = s.get("text", "")

                            st.markdown(f"**Source {i} — Page {page}**")
                            st.write(text[:500] + ("..." if len(text) > 500 else ""))
                            st.divider()

                # Store assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"⚠️ Failed to call API: {e}")



