import streamlit as st
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# ---------- App Config ----------
st.set_page_config(page_title="GasGridBot", page_icon="💡")
# --- Bot mode toggle (goes near the top, before Sidebar) ---
mode = st.sidebar.radio(
    "Select Bot Mode:",
    ("GasGridBot (RAG Search)", "General GPT-3.5 Chat")
)

# Optional dynamic header/subtitle
if mode == "GasGridBot (RAG Search)":
    st.markdown("**Mode:** Domain-specific RAG over Hydrotest, Compliance, Corrosion & Methane docs.")
else:
    st.markdown("**Mode:** Open-domain GPT-3.5 (not grounded in your documents).")
    st.info("⚠️ This mode does not use your uploaded PDFs or Cognitive Search.")

st.title("💡 GasGridBot by Akshay Sharma")
st.caption("AI assistant for Midstream Natural Gas Utilites (RAG POC)")

# ---------- Read secrets from Streamlit Cloud ----------
cfg = st.secrets

# OpenAI (Azure) setup
openai.api_type = "azure"
openai.api_base = cfg["AZURE_OPENAI_ENDPOINT"]
openai.api_key = cfg["AZURE_OPENAI_KEY"]
openai.api_version = cfg.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

AZURE_EMBEDDING_DEPLOYMENT = cfg["AZURE_EMBEDDING_DEPLOYMENT"]
AZURE_CHAT_DEPLOYMENT = cfg["AZURE_CHAT_DEPLOYMENT"]

# Cognitive Search setup
@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=cfg["AZURE_SEARCH_ENDPOINT"],
        index_name=cfg["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(cfg["AZURE_SEARCH_KEY"])
    )

search_client = get_search_client()

# ---------- Helper: retrieve top-K context ----------
def retrieve_context(query: str, top_k: int = 3):
    emb = openai.Embedding.create(input=query, engine=AZURE_EMBEDDING_DEPLOYMENT)["data"][0]["embedding"]
    results = search_client.search(
        vector_queries=[VectorizedQuery(vector=emb, fields="embedding")],
        top=top_k,
        select=["content", "source"]
    )
    contexts, sources = [], []
    for r in results:
        content = r.get("content", "")
        source = r.get("source", "")
        if content:
            contexts.append(content)
            sources.append(source)
    return "\n\n".join(contexts), sources
def answer_with_context(user_query: str, context_text: str, temperature=0, max_tokens=600, system_hint=None):
    sys_msg = system_hint or (
        "You are GasGridBot, an assistant that answers ONLY from the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    resp = openai.ChatCompletion.create(
        engine=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp["choices"][0]["message"]["content"]

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### About")
    st.write("GasGridBot uses RAG over Hydrotest PDFs in Azure Cognitive Search.")
    st.markdown("**Try:**")
    st.markdown("- What was the max hydrotest pressure and hold time?")
    st.markdown("- Any sections below allowable limit and corrective actions?")
    st.markdown("- Compare results for Line A vs Line B.")
    st.markdown("- Did we comply with ASME B31.8S stabilization/monitoring?")
    st.markdown("- What anomalies were observed and possible causes?")

    # 🔒 Debug Mode
    if st.checkbox("🔑 Debug mode (private)", value=False):
        st.subheader("Secrets Validation")
        try:
            st.write("AZURE_OPENAI_ENDPOINT:", cfg["AZURE_OPENAI_ENDPOINT"])
            st.write("AZURE_OPENAI_KEY:", "✅ Found (hidden)")
            st.write("AZURE_OPENAI_API_VERSION:", cfg["AZURE_OPENAI_API_VERSION"])
            st.write("AZURE_EMBEDDING_DEPLOYMENT:", cfg["AZURE_EMBEDDING_DEPLOYMENT"])
            st.write("AZURE_CHAT_DEPLOYMENT:", cfg["AZURE_CHAT_DEPLOYMENT"])
            st.write("AZURE_SEARCH_ENDPOINT:", cfg["AZURE_SEARCH_ENDPOINT"])
            st.write("AZURE_SEARCH_KEY:", "✅ Found (hidden)")
            st.write("AZURE_SEARCH_INDEX_NAME:", cfg["AZURE_SEARCH_INDEX_NAME"])
            st.success("🎉 Secrets loaded successfully!")

            if st.button("Test Azure Connection"):
                try:
                    # Test OpenAI
                    test_resp = openai.ChatCompletion.create(
                        engine=AZURE_CHAT_DEPLOYMENT,
                        messages=[{"role": "user", "content": "Hello, are you working?"}],
                        max_tokens=20
                    )
                    st.write("✅ OpenAI test:", test_resp["choices"][0]["message"]["content"])

                    # Test Search
                    search_results = search_client.search(search_text="test", top=1)
                    first_result = next(iter(search_results), None)
                    if first_result:
                        st.write("✅ Cognitive Search test: found sample doc")
                    else:
                        st.warning("⚠️ Cognitive Search connected but no docs found")
                except Exception as e:
                    st.error("❌ Connection test failed")
                    st.exception(e)

        except KeyError as e:
            st.error(f"❌ Missing secret: {e}")

# ---------- Chat UI (dual-mode) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

# Render existing history
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Input
user_query = st.chat_input("Ask a question...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            if mode == "GasGridBot (RAG Search)":
                # 1) Retrieve context from Cognitive Search
                context_text, sources = retrieve_context(user_query, top_k=3)

                # Guardrail: no context found
                if not context_text.strip():
                    bot_reply = "I don't have relevant context in the indexed documents to answer this."
                    st.warning("No relevant context found in your indexed documents.")
                else:
                    # 2) Ask GPT with context
                    bot_reply = answer_with_context(user_query, context_text, temperature=0, max_tokens=600)

                st.markdown(bot_reply)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- {s}")

            else:  # General GPT-3.5 Chat (no RAG)
                response = openai.ChatCompletion.create(
                    engine=AZURE_CHAT_DEPLOYMENT,  # e.g. "gpt-35-turbo"
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.3,
                    max_tokens=600
                )
                bot_reply = response["choices"][0]["message"]["content"]
                st.markdown(bot_reply)

        except Exception as e:
            st.error("Something went wrong. Check your secrets and index configuration.")
            st.exception(e)
            bot_reply = "Error."

    # Save bot reply to history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

