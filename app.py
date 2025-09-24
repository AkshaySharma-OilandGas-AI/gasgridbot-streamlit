import streamlit as st
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# ---------- App Config ----------
st.set_page_config(page_title="GasGridBot", page_icon="💡")
st.title("💡 GasGridBot")
st.caption("AI assistant for Midstream Natural Gas Utilites (RAG POC)")

# ---------- Read secrets from Streamlit Cloud ----------
cfg = st.secrets  # all keys live in Streamlit Cloud -> App -> Settings -> Secrets

# OpenAI (Azure) setup
openai.api_type = "azure"
openai.api_base = cfg["AZURE_OPENAI_ENDPOINT"]          # e.g. https://<your-aoai>.openai.azure.com/
openai.api_key = cfg["AZURE_OPENAI_KEY"]
openai.api_version = cfg.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

AZURE_EMBEDDING_DEPLOYMENT = cfg["AZURE_EMBEDDING_DEPLOYMENT"]  # e.g. "text-embedding-ada-002"
AZURE_CHAT_DEPLOYMENT = cfg["AZURE_CHAT_DEPLOYMENT"]            # e.g. "gpt-35-turbo"

# Cognitive Search setup (cached so it’s created once per session)
@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=cfg["AZURE_SEARCH_ENDPOINT"],              # e.g. https://<your-search>.search.windows.net
        index_name=cfg["AZURE_SEARCH_INDEX_NAME"],          # e.g. "hydrotest-index"
        credential=AzureKeyCredential(cfg["AZURE_SEARCH_KEY"])
    )

search_client = get_search_client()

# ---------- Helper: retrieve top-K context ----------
def retrieve_context(query: str, top_k: int = 3):
    # Embed query
    emb = openai.Embedding.create(input=query, engine=AZURE_EMBEDDING_DEPLOYMENT)["data"][0]["embedding"]
    # Vector search
    results = search_client.search(
        vector_queries=[VectorizedQuery(vector=emb, fields="embedding")],
        top=top_k,
        select=["content", "source"]
    )
    contexts = []
    sources = []
    for r in results:
        content = r.get("content", "")
        source = r.get("source", "")
        if content:
            contexts.append(content)
            sources.append(source)
    return "\n\n".join(contexts), sources

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role": "...", "content": "..."}

# ---------- Sidebar (about + sample queries) ----------
with st.sidebar:
    st.markdown("### About")
    st.write("GasGridBot uses RAG over your Hydrotest PDFs in Azure Cognitive Search.")
    st.markdown("**Try:**")
    st.markdown("- What was the max hydrotest pressure and hold time?")
    st.markdown("- Any sections below allowable limit and corrective actions?")
    st.markdown("- Compare results for Line A vs Line B.")
    st.markdown("- Did we comply with ASME B31.8S stabilization/monitoring?")
    st.markdown("- What anomalies were observed and possible causes?")

# ---------- Chat UI ----------
user_query = st.chat_input("Ask GasGridBot about Hydrotest reports…")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    try:
        # 1) Retrieve context
        context_text, sources = retrieve_context(user_query, top_k=3)

        # 2) Build messages (include minimal history for vibe, but ground on context)
        messages = [{"role": "system",
                     "content": "You are GasGridBot, an assistant that answers ONLY from the provided Hydrotest report context. "
                                "If the answer is not in context, say you don't know."}]
        for turn in st.session_state.history[-6:]:  # keep last few turns
            messages.append(turn)
        messages.append({"role": "user",
                         "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"})

        # 3) Ask GPT
        resp = openai.ChatCompletion.create(
            engine=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=600
        )
        bot_reply = resp["choices"][0]["message"]["content"]

        # 4) Render answer
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
