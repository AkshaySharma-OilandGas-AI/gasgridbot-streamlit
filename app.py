import streamlit as st
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# ---------- App Config ----------
st.set_page_config(page_title="GasGridBot", page_icon="💡")
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

# ---------- Chat UI ----------
user_query = st.chat_input("Ask GasGridBot about Hydrotest reports…")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    try:
        # 1) Retrieve context
        context_text, sources = retrieve_context(user_query, top_k=3)

        # 2) Build conversation
        messages = [{"role": "system",
                     "content": "You are GasGridBot, an assistant that answers ONLY from Hydrotest report context. "
                                "If answer not in context, say you don't know."}]
        for turn in st.session_state.history[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"})

        # 3) Ask GPT
        resp = openai.ChatCompletion.create(
            engine=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=600
        )
        bot_reply = resp["choices"][0]["message"]["content"]

        # 4) Render
        with st.chat_message("assistant"):
            st.write(bot_reply)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s}")

        # Save to history
        st.session_state.history.append({"role": "user", "content": user_query})
        st.session_state.history.append({"role": "assistant", "content": bot_reply})

    except Exception as e:
        st.error("Something went wrong. Check secrets and index configuration.")
        st.exception(e)
