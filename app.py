import os
import re
import json
import random
import base64
import datetime
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from dotenv import load_dotenv

# =====================
# PAGE CONFIG — MUST be first Streamlit call
# =====================
st.set_page_config(
    page_title="🇵🇸 Palestine RAG Chatbot",
    page_icon="🇵🇸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================
# SESSION STATE — initialize ALL keys upfront
# =====================
defaults = {
    "chat_messages": [],
    "timeline_result": None,
    "index_loaded": False,
    "doc_list": [],
    "chunk_stats": {},
    "total_chunks": 0,
    "llm_client": None,
    "tts_lang": "en",
    "translate_history": [],
    "export_ready": False,
    "model_a": "gpt-oss-120b",
    "model_b": "gpt-oss-70b",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================
# LOAD ENV
# =====================
load_dotenv()

# =====================
# CSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap');
#MainMenu,footer,header{visibility:hidden;}
html,body,.stApp{background:#0e0e0e;color:#e5e5e5;font-family:'Source Sans 3',sans-serif;}
h1,h2,h3{font-family:'Amiri',serif;color:#f5f0e8;}
.stTabs [data-baseweb="tab-list"]{background:#141414;border-bottom:2px solid #1e6e3a;gap:0;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#999;font-size:0.78rem;font-weight:600;letter-spacing:.04em;padding:.55rem .85rem;border:none;text-transform:uppercase;}
.stTabs [aria-selected="true"]{background:#1e6e3a!important;color:#fff!important;border-radius:4px 4px 0 0;}
.stTabs [data-baseweb="tab-panel"]{background:#0e0e0e;padding-top:1.5rem;}
.card{background:#161616;border:1px solid #252525;border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;}
.stChatMessage{background:#161616!important;border:1px solid #252525;border-radius:10px;margin-bottom:.5rem;}
textarea,.stTextInput input{background:#161616!important;color:#e5e5e5!important;border:1px solid #2a2a2a!important;border-radius:8px!important;}
.stButton>button{background:#1e6e3a;color:#fff;border:none;border-radius:6px;font-weight:600;padding:.4rem 1.2rem;transition:background .2s;}
.stButton>button:hover{background:#27904d;color:#fff;}
[data-testid="stMetric"]{background:#161616;border:1px solid #252525;border-radius:8px;padding:.8rem 1rem;}
[data-testid="stMetricLabel"]{color:#888!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#4caf70!important;font-size:1.6rem;}
.timeline-entry{border-left:3px solid #1e6e3a;padding-left:1rem;margin-bottom:1.2rem;}
.timeline-year{color:#4caf70;font-weight:700;font-size:1rem;}
.source-chip{display:inline-block;background:#1a3a26;color:#4caf70;border:1px solid #1e6e3a;border-radius:20px;padding:2px 10px;font-size:.75rem;margin:2px;}
.flag-bar{height:5px;background:linear-gradient(to right,#000 33%,#fff 33%,#fff 66%,#1e6e3a 66%);margin-bottom:.5rem;border-radius:2px;}
.green-divider{border:none;border-top:1px solid #1e6e3a;margin:1rem 0;}
.feature-badge{display:inline-block;background:#1a3a26;color:#4caf70;border:1px solid #1e6e3a;border-radius:4px;padding:2px 8px;font-size:.7rem;font-weight:700;margin-left:6px;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0e0e0e;}
::-webkit-scrollbar-thumb{background:#1e6e3a;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# =====================
# SAFE PIPELINE IMPORT
# =====================
try:
    import pdf_pipeline
    from pdf_pipeline import initialize, retrieve
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    pdf_pipeline = None

# =====================
# SYSTEM INIT
# =====================
@st.cache_resource(show_spinner="📚 Loading document index...")
def load_system():
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="http://app.ai-grid.io:4000/v1"
    )
    if PIPELINE_AVAILABLE:
        try:
            initialize("data")
        except Exception as e:
            st.warning(f"Index init warning: {e}")
    return client

llm_client = load_system()

# =====================
# METADATA HELPERS — robust to different pdf_pipeline implementations
# =====================
def _get_metadata():
    """Try multiple attribute names that pdf_pipeline might use."""
    if not PIPELINE_AVAILABLE:
        return []
    for attr in ["metadata", "chunks_metadata", "meta", "all_metadata", "chunk_meta"]:
        val = getattr(pdf_pipeline, attr, None)
        if val:
            return val
    # Try to reconstruct from index if possible
    return []

def get_doc_list():
    meta = _get_metadata()
    if meta:
        docs = sorted(set(m.get("source", m.get("file", m.get("doc", ""))) for m in meta if m))
        return [d for d in docs if d]
    # Fallback: scan data/ folder
    data_dir = "data"
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
        if files:
            return sorted(files)
    return []

def get_chunk_stats():
    meta = _get_metadata()
    if meta:
        counts = {}
        for m in meta:
            src = m.get("source", m.get("file", m.get("doc", "unknown")))
            counts[src] = counts.get(src, 0) + 1
        return counts
    return {}

def safe_retrieve(query, k=5):
    """Retrieve with fallback."""
    if not PIPELINE_AVAILABLE:
        return []
    try:
        return retrieve(query, k)
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

# =====================
# LLM HELPER
# =====================
def llm_call(prompt, system=None, max_tokens=1200, model=None):
    if model is None:
        model = "gpt-oss-120b"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        res = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Error: {e}"

def rag_answer(query, k=5, model=None):
    retrieved = safe_retrieve(query, k)
    if not retrieved:
        return llm_call(f"Answer based on general knowledge: {query}", max_tokens=800, model=model), []
    context = ""
    sources = []
    for text, meta in retrieved:
        src = meta.get("source", meta.get("file", "unknown"))
        page = meta.get("page", "?")
        context += f"(Source: {src} | Page {page}):\n{text}\n\n"
        sources.append({"source": src, "page": page})
    prompt = (
        "Answer ONLY using the context below.\n"
        "If not found, say: Not found in documents.\n"
        "Always cite the source document and page number.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )
    return llm_call(prompt, model=model), sources

# =====================
# HEADER
# =====================
st.markdown('<div class="flag-bar"></div>', unsafe_allow_html=True)
st.markdown("# 🇵🇸 Palestine RAG Chatbot")
st.markdown("*Retrieval-Augmented Generation · FAISS · OCR · 15 Documents*")
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# =====================
# TABS — 10 mandatory + 5 bonus
# =====================
tab_labels = [
    "💬 Smart Chat",
    "🔍 Discourse",
    "📊 Compare",
    "📄 Summary",
    "🗺️ Map",
    "📅 Timeline",
    "☁️ Word Cloud",
    "📈 Statistics",
    "📤 Upload PDF",
    "ℹ️ About",
    "🎙️ Voice Input",
    "🌐 Translate",
    "📊 Analytics",
    "💾 Export Chat",
    "🤖 Multi-Model",
]
tabs = st.tabs(tab_labels)

# ═══════════════════════════════════════════════
# TAB 1 — SMART CHAT
# ═══════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 💬 Smart Chat")
    st.markdown("RAG chatbot with automatic source and page number citation.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl3:
        if st.button("🗑️ Clear", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                chips = " ".join(
                    f'<span class="source-chip">📄 {s["source"]} p.{s["page"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(chips, unsafe_allow_html=True)

    user_input = st.chat_input("Ask about Palestine, Gaza, history, human rights...", key="chat_input")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("🔎 Retrieving and generating..."):
                answer, sources = rag_answer(user_input)
            st.markdown(answer)
            if sources:
                chips = " ".join(
                    f'<span class="source-chip">📄 {s["source"]} p.{s["page"]}</span>'
                    for s in sources
                )
                st.markdown(chips, unsafe_allow_html=True)
        st.session_state.chat_messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })

# ═══════════════════════════════════════════════
# TAB 2 — DISCOURSE ANALYSIS
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🔍 Discourse Analysis")
    st.markdown("Detect bias, propaganda indicators, and text orientation.")

    da_mode = st.radio("Input mode", ["Enter text manually", "Retrieve from documents"],
                       horizontal=True, key="da_mode")

    if da_mode == "Enter text manually":
        da_text = st.text_area("Paste text to analyze", height=200,
                               placeholder="Paste a news article, speech, or excerpt...", key="da_text")
    else:
        da_topic = st.text_input("Topic to retrieve", placeholder="e.g. Israeli settlements", key="da_topic")
        da_text = ""
        if da_topic:
            with st.spinner("Retrieving..."):
                retrieved = safe_retrieve(da_topic, 3)
                da_text = "\n\n".join(t for t, _ in retrieved) if retrieved else ""
            if da_text:
                st.text_area("Retrieved text", value=da_text, height=120, key="da_preview")

    if st.button("🔍 Analyze", key="da_btn"):
        if not da_text.strip():
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("Analyzing discourse..."):
                result = llm_call(
                    f"Analyze the following text for:\n"
                    f"1. **Bias indicators** (loaded language, omissions, framing)\n"
                    f"2. **Propaganda techniques** (appeal to emotion, scapegoating, repetition)\n"
                    f"3. **Political/ideological orientation**\n"
                    f"4. **Overall assessment** (neutral/pro-Palestinian/pro-Israeli/mixed)\n\n"
                    f"Be specific, cite exact phrases from the text.\n\nText:\n{da_text[:3000]}",
                    system="You are an expert discourse analyst specializing in Middle East media and political texts."
                )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3 — COMPARE DOCUMENTS
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📊 Compare Documents")
    st.markdown("Side-by-side comparison of two topics or documents.")

    col_a, col_b = st.columns(2)
    with col_a:
        topic_a = st.text_input("Topic A", placeholder="e.g. Oslo Accords", key="cmp_a")
    with col_b:
        topic_b = st.text_input("Topic B", placeholder="e.g. Camp David Summit", key="cmp_b")
    aspect = st.text_input("Comparison aspect (optional)",
                           placeholder="e.g. impact on Palestinian sovereignty", key="cmp_aspect")

    if st.button("⚖️ Compare", key="cmp_btn"):
        if not topic_a or not topic_b:
            st.warning("Please enter both topics.")
        else:
            with st.spinner("Comparing..."):
                ret_a = safe_retrieve(topic_a, 3)
                ret_b = safe_retrieve(topic_b, 3)
                ctx_a = "\n\n".join(
                    f"(Source: {m.get('source','?')} p.{m.get('page','?')}):\n{t}" for t, m in ret_a
                ) if ret_a else f"No indexed chunks found for '{topic_a}'."
                ctx_b = "\n\n".join(
                    f"(Source: {m.get('source','?')} p.{m.get('page','?')}):\n{t}" for t, m in ret_b
                ) if ret_b else f"No indexed chunks found for '{topic_b}'."
                aspect_str = f" Focus on: {aspect}." if aspect else ""
                result = llm_call(
                    f"Compare '{topic_a}' and '{topic_b}'.{aspect_str}\n"
                    f"Structure: **Similarities**, **Differences**, **Key Tensions**, **Conclusion**.\n"
                    f"Cite sources.\n\n=== {topic_a} ===\n{ctx_a}\n\n=== {topic_b} ===\n{ctx_b}",
                    system="You are a political analyst specializing in the Palestinian-Israeli conflict.",
                    max_tokens=1500,
                )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 📘 {topic_a}")
                for t, m in ret_a:
                    st.caption(f"📄 {m.get('source','?')} — p.{m.get('page','?')}")
                    st.markdown(f'<div class="card">{t[:350]}…</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"### 📗 {topic_b}")
                for t, m in ret_b:
                    st.caption(f"📄 {m.get('source','?')} — p.{m.get('page','?')}")
                    st.markdown(f'<div class="card">{t[:350]}…</div>', unsafe_allow_html=True)
            st.markdown("### ⚖️ Analysis")
            st.markdown(f'<div class="card">{result}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 4 — DOCUMENT SUMMARY
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 📄 Document Summary")
    st.markdown("Auto-summarize any of the indexed documents.")

    # Refresh doc list every render
    doc_list = get_doc_list()
    if not doc_list:
        st.warning("⚠️ No documents found. Make sure PDFs are in the `data/` folder and the index is built.")
        st.info("💡 Go to the **Upload PDF** tab to add documents, or check that `initialize('data')` ran successfully.")
    else:
        selected_doc = st.selectbox("Select a document", doc_list, key="sum_doc")
        summary_style = st.radio(
            "Summary style",
            ["Concise (3–5 bullets)", "Detailed (3 paragraphs)", "Key arguments only"],
            horizontal=True, key="sum_style"
        )
        if st.button("📄 Summarize", key="sum_btn"):
            with st.spinner(f"Summarizing {selected_doc}..."):
                retrieved = safe_retrieve(selected_doc, 8)
                if retrieved:
                    doc_chunks = [(t, m) for t, m in retrieved
                                  if selected_doc.lower() in str(m.get("source", m.get("file", ""))).lower()]
                    if not doc_chunks:
                        doc_chunks = retrieved
                    context = "\n\n".join(f"(Page {m.get('page','?')}):\n{t}" for t, m in doc_chunks)
                else:
                    context = f"Document: {selected_doc}. No chunks retrieved."
                    doc_chunks = []

                style_map = {
                    "Concise (3–5 bullets)": "Provide a concise summary as 3–5 bullet points.",
                    "Detailed (3 paragraphs)": "Provide a detailed summary in 3 well-structured paragraphs.",
                    "Key arguments only": "Extract and list only the key arguments and claims.",
                }
                result = llm_call(
                    f"Document: {selected_doc}\n\n{style_map[summary_style]}\n"
                    f"Include page references. Use ONLY the content below.\n\nContent:\n{context[:4000]}",
                    system="You are an expert summarizer of political and historical documents.",
                    max_tokens=1000,
                )
            st.markdown(f"### Summary: *{selected_doc}*")
            st.markdown(f'<div class="card">{result}</div>', unsafe_allow_html=True)
            st.caption(f"Based on {len(doc_chunks)} retrieved chunks.")

# ═══════════════════════════════════════════════
# TAB 5 — INTERACTIVE MAP
# ═══════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🗺️ Interactive Map")
    st.markdown("Palestinian locations with historical context. Click any marker.")

    locations = [
        ("Gaza City", 31.5017, 34.4668, "Capital of the Gaza Strip. Under blockade since 2007. Population ~750,000. Major site of recurring military conflict."),
        ("Jerusalem (Al-Quds)", 31.7683, 35.2137, "Holy city. East Jerusalem occupied since 1967. Home to Al-Aqsa Mosque and the Church of the Holy Sepulchre."),
        ("Ramallah", 31.8996, 35.2042, "De facto administrative capital of the Palestinian Authority. Located in the West Bank."),
        ("Hebron (Al-Khalil)", 31.5326, 35.0998, "Largest city in the West Bank. Divided city with Israeli settlers in the center."),
        ("Nablus", 32.2211, 35.2544, "Economic capital of Palestine. Frequent Israeli military incursions."),
        ("Rafah", 31.2870, 34.2473, "Southern tip of Gaza. Over 1M displaced persons sheltered here in 2024."),
        ("Jenin", 32.4607, 35.2961, "Refugee camp and city. Site of major Israeli operations in 2002 and 2023."),
        ("Bethlehem", 31.7054, 35.2024, "Birthplace of Jesus. Surrounded by the Israeli separation wall."),
        ("Jericho", 31.8522, 35.4444, "One of the oldest cities in the world. Under Palestinian Authority control."),
        ("Khan Yunis", 31.3452, 34.3063, "Second largest city in Gaza. Intense military operations in 2023–2024."),
        ("Haifa", 32.7940, 34.9896, "Historic Palestinian port city, now in Israel. Site of the Nakba expulsions in 1948."),
        ("Acre (Akka)", 32.9236, 35.0680, "Ancient port city. UNESCO World Heritage Site."),
    ]

    markers_js = ""
    for name, lat, lon, info in locations:
        safe_info = info.replace("'", "\\'").replace('"', '\\"')
        safe_name = name.replace("'", "\\'")
        markers_js += f"L.marker([{lat},{lon}]).addTo(map).bindPopup('<b>{safe_name}</b><br><br>{safe_info}',{{maxWidth:280}});\n"

    map_html = f"""<!DOCTYPE html><html><head>
    <meta charset="utf-8"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body{{margin:0;background:#0e0e0e;}}
        #map{{height:520px;width:100%;border-radius:10px;}}
        .leaflet-popup-content-wrapper{{background:#1a1a1a;color:#e5e5e5;border:1px solid #1e6e3a;border-radius:8px;}}
        .leaflet-popup-tip{{background:#1a1a1a;}}
        .leaflet-popup-content b{{color:#4caf70;font-size:1.05rem;}}
    </style></head><body>
    <div id="map"></div>
    <script>
    var map=L.map('map').setView([31.9,35.2],7);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'© OpenStreetMap',maxZoom:18}}).addTo(map);
    {markers_js}
    </script></body></html>"""

    components.html(map_html, height=540)
    st.caption("📍 12 locations. Click markers for historical context.")

# ═══════════════════════════════════════════════
# TAB 6 — HISTORICAL TIMELINE
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📅 Historical Timeline")

    tl_topic = st.text_input(
        "Topic (leave blank for general history)",
        placeholder="e.g. Gaza blockade, Oslo process",
        key="tl_topic"
    )

    if st.button("⚡ Generate Timeline", key="tl_btn"):
        topic_str = tl_topic.strip() if tl_topic.strip() else "Palestinian history from 1900 to present"
        with st.spinner("Building timeline..."):
            ctx = ""
            if PIPELINE_AVAILABLE and tl_topic.strip():
                retrieved = safe_retrieve(tl_topic, 6)
                if retrieved:
                    ctx = "Use this context as primary source:\n\n" + "\n\n".join(
                        f"(Source: {m.get('source','?')} p.{m.get('page','?')}):\n{t}"
                        for t, m in retrieved
                    ) + "\n\n"
            result = llm_call(
                f"{ctx}Create a chronological timeline for: {topic_str}.\n"
                f"Format EXACTLY as: YEAR: TITLE | DESCRIPTION\n"
                f"List at least 12 events ordered by year.",
                system="You are a historian specializing in Palestinian and Middle Eastern history.",
                max_tokens=1500,
            )
        st.session_state.timeline_result = result

    if st.session_state.timeline_result:
        for line in st.session_state.timeline_result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if "|" in line and ":" in line.split("|")[0]:
                parts = line.split("|", 1)
                year_title = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                st.markdown(f"""
                <div class="timeline-entry">
                    <div class="timeline-year">{year_title}</div>
                    <div style="color:#ccc;margin-top:4px;">{desc}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:#bbb;margin-bottom:.4rem;">{line}</div>', unsafe_allow_html=True)
    else:
        static = [
            ("1917", "Balfour Declaration", "Britain expresses support for a Jewish homeland in Palestine."),
            ("1948", "Nakba — Israeli Statehood", "700,000+ Palestinians expelled or flee. State of Israel declared."),
            ("1967", "Six-Day War", "Israel occupies West Bank, Gaza, Sinai, and Golan Heights."),
            ("1973", "Yom Kippur War", "Egypt and Syria launch surprise attack. UN-brokered ceasefire."),
            ("1987", "First Intifada", "Palestinian uprising against Israeli occupation begins."),
            ("1993", "Oslo Accords", "PLO and Israel sign Declaration of Principles in Washington D.C."),
            ("2000", "Second Intifada", "Collapse of Camp David talks triggers new wave of violence."),
            ("2005", "Gaza Disengagement", "Israel withdraws settlements and military from Gaza Strip."),
            ("2007", "Gaza Blockade Begins", "Israel and Egypt impose land, air, and sea blockade on Gaza."),
            ("2014", "Operation Protective Edge", "50-day war on Gaza. Over 2,200 Palestinians killed."),
            ("2023", "October 7 & War on Gaza", "Hamas attacks. Israel launches major military campaign on Gaza."),
            ("2024", "ICJ Genocide Case", "South Africa files genocide case against Israel at the ICJ."),
        ]
        for year, title, desc in static:
            st.markdown(f"""
            <div class="timeline-entry">
                <div class="timeline-year">{year} — {title}</div>
                <div style="color:#ccc;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 7 — WORD CLOUD
# ═══════════════════════════════════════════════
with tabs[6]:
    st.markdown("## ☁️ Word Cloud")
    st.markdown("Word frequency visualization per document.")

    doc_list_wc = get_doc_list()
    if not doc_list_wc:
        st.warning("⚠️ No documents found. Upload PDFs and build the index first.")
    else:
        wc_doc = st.selectbox("Select document", doc_list_wc, key="wc_doc")
        top_n = st.slider("Top N words", 20, 80, 40, key="wc_n")

        if st.button("☁️ Generate Word Cloud", key="wc_btn"):
            with st.spinner("Processing text..."):
                retrieved = safe_retrieve(wc_doc, 10)
                all_text = " ".join(t for t, _ in retrieved).lower() if retrieved else ""

                if not all_text:
                    st.warning("Could not retrieve text for this document.")
                else:
                    stopwords = {
                        "the","a","an","and","or","but","in","on","at","to","for","of","with",
                        "is","was","are","were","be","been","being","have","has","had","do","does",
                        "did","will","would","could","should","may","might","shall","can","this",
                        "that","these","those","it","its","by","from","as","into","through","also",
                        "which","who","whom","what","when","where","their","they","them","he","she",
                        "his","her","we","our","us","you","your","not","no","so","yet","both","more",
                        "most","other","some","such","than","very","just","because","if","then",
                        "there","here","all","any","between","after","before","about","over","under",
                        "i","my","me","one","two","new","well","made","first","second","since","upon",
                        "within","page","source","document","said","each","only","even","while","each",
                    }
                    words = re.findall(r'\b[a-z]{4,}\b', all_text)
                    freq = {}
                    for w in words:
                        if w not in stopwords:
                            freq[w] = freq.get(w, 0) + 1

                    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

                    if sorted_words:
                        max_f = sorted_words[0][1]
                        min_f = sorted_words[-1][1]
                        colors = ["#4caf70","#81c995","#a8d5b5","#ffffff","#f5f0e8","#c9a84c","#e8c97a"]
                        random.seed(42)
                        cols_n = 7
                        svg_words = ""
                        for i, (word, f) in enumerate(sorted_words):
                            size = 11 + int(26 * (f - min_f) / max(max_f - min_f, 1))
                            color = random.choice(colors)
                            x = 70 + (i % cols_n) * 120
                            y = 50 + (i // cols_n) * 52
                            svg_words += f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-family="Georgia" opacity=".9">{word}</text>\n'
                        rows = (len(sorted_words) // cols_n) + 2
                        h = rows * 52 + 60
                        svg = (f'<svg width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" '
                               f'style="background:#161616;border-radius:10px;border:1px solid #252525;">'
                               f'<rect width="100%" height="100%" fill="#161616" rx="10"/>{svg_words}</svg>')
                        st.markdown(svg, unsafe_allow_html=True)
                        st.markdown("### 📊 Top 15 Frequencies")
                        st.bar_chart(dict(sorted_words[:15]))
                    else:
                        st.warning("No words extracted.")

# ═══════════════════════════════════════════════
# TAB 8 — STATISTICS
# ═══════════════════════════════════════════════
with tabs[7]:
    st.markdown("## 📈 Statistics")
    st.markdown("Chunk count, distribution per document, coverage metrics, and document list.")

    # Always recompute
    chunk_stats = get_chunk_stats()
    total_chunks = sum(chunk_stats.values())
    total_docs = len(chunk_stats)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Chunks", total_chunks if total_chunks else "—")
    c2.metric("📄 Documents Indexed", total_docs if total_docs else "—")
    c3.metric("🔍 Retrieval K", "5")
    c4.metric("🤖 Model", "gpt-oss-120b")

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    if chunk_stats:
        st.markdown("### Chunk Distribution per Document")
        st.bar_chart(chunk_stats)
        st.markdown("### 📋 Document Inventory")
        for i, (doc, count) in enumerate(sorted(chunk_stats.items(), key=lambda x: x[1], reverse=True), 1):
            pct = count / total_chunks * 100
            st.markdown(
                f'<div class="card" style="padding:.6rem 1rem;margin-bottom:.3rem;">'
                f'<b style="color:#4caf70;">#{i}</b> &nbsp;{doc}'
                f'<span style="float:right;color:#888;">{count} chunks &nbsp;|&nbsp; {pct:.1f}%</span></div>',
                unsafe_allow_html=True
            )
    else:
        # Fallback: show files in data/ folder
        data_dir = "data"
        pdf_files = []
        if os.path.isdir(data_dir):
            pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

        if pdf_files:
            st.info(f"✅ Found {len(pdf_files)} PDF(s) in `data/` folder, but metadata not yet in memory. The index may still be loading.")
            for i, f in enumerate(sorted(pdf_files), 1):
                size = os.path.getsize(os.path.join(data_dir, f)) / 1024
                st.markdown(
                    f'<div class="card" style="padding:.5rem 1rem;margin-bottom:.3rem;">'
                    f'<b style="color:#4caf70;">#{i}</b> &nbsp;{f}'
                    f'<span style="float:right;color:#888;">{size:.1f} KB</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No documents indexed yet. Upload PDFs via the **Upload PDF** tab.")
            st.markdown("### Expected Corpus (15 documents)")
            placeholder_docs = [
                "UN Report on Gaza 2023", "Oslo Accords Full Text",
                "ICJ Ruling South Africa vs Israel", "Human Rights Watch Report 2022",
                "Amnesty International Palestine 2023", "UNRWA Statistics 2023",
                "Palestinian Central Bureau of Statistics", "B'Tselem Settler Violence Report",
                "WHO Gaza Health Update 2024", "OCHA Humanitarian Snapshot",
                "Geneva Conventions Commentary", "PLO Charter",
                "Hamas Charter 2017", "Abraham Accords Text", "Arab Peace Initiative 2002",
            ]
            for i, doc in enumerate(placeholder_docs, 1):
                st.markdown(
                    f'<div class="card" style="padding:.5rem 1rem;margin-bottom:.3rem;">'
                    f'<b style="color:#4caf70;">#{i}</b> &nbsp;{doc}</div>',
                    unsafe_allow_html=True
                )

# ═══════════════════════════════════════════════
# TAB 9 — UPLOAD PDF
# ═══════════════════════════════════════════════
with tabs[8]:
    st.markdown("## 📤 Upload PDF")
    st.markdown("Upload a new PDF and query it instantly without restarting.")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="upload_pdf")

    if uploaded is not None:
        st.success(f"✅ Received: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        if st.button("⚙️ Index This Document", key="index_btn"):
            with st.spinner("Saving and indexing..."):
                try:
                    os.makedirs("data", exist_ok=True)
                    save_path = os.path.join("data", uploaded.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    if PIPELINE_AVAILABLE:
                        # Clear cache so load_system re-runs
                        load_system.clear()
                        initialize("data")
                        st.cache_resource.clear()
                        st.success(f"✅ **{uploaded.name}** indexed successfully!")
                        st.info("🔄 The index has been rebuilt. All tabs now include this document.")
                        st.rerun()
                    else:
                        st.error("❌ pdf_pipeline module not found. Cannot index.")
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown("### 🔎 Query Uploaded Document")
    up_query = st.text_input("Ask a question about your document",
                             placeholder="e.g. What are the main findings?", key="up_query")

    if st.button("🔎 Search", key="up_search_btn") and up_query:
        with st.spinner("Searching..."):
            answer, sources = rag_answer(up_query)
        st.markdown(f'<div class="card">{answer}</div>', unsafe_allow_html=True)
        for s in sources:
            st.caption(f"📄 {s['source']} — page {s['page']}")

    # Show indexed files
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown("### 📂 Currently Indexed Files")
    if os.path.isdir("data"):
        pdfs = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]
        if pdfs:
            for f in sorted(pdfs):
                size = os.path.getsize(os.path.join("data", f)) / 1024
                st.markdown(
                    f'<div class="card" style="padding:.5rem 1rem;margin-bottom:.3rem;">'
                    f'📄 {f} <span style="float:right;color:#888;">{size:.1f} KB</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No PDF files in data/ folder yet.")
    else:
        st.info("data/ folder not found.")

# ═══════════════════════════════════════════════
# TAB 10 — ABOUT
# ═══════════════════════════════════════════════
with tabs[9]:
    st.markdown("## ℹ️ About the Project")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
### 🎯 Mission
This RAG chatbot provides accurate, document-grounded answers about the Palestinian cause,
history, and humanitarian situation — sourced from 15 authoritative documents (1917–2024).
---
### 🏗️ Architecture
**Ingestion Pipeline**
- PDFs processed with OCR (Tesseract / pdfplumber)
- Text chunked with overlap for context preservation
- Chunks embedded via API embeddings
**Retrieval**
- FAISS vector index — fast similarity search
- Top-K chunk retrieval with source + page metadata
**Generation**
- LLM (gpt-oss-120b) via AI Grid API
- Answers strictly grounded in retrieved context
**Frontend**
- Streamlit 15-tab interface
- Interactive Leaflet.js map
- Dynamic word clouds and statistics
---
### 📚 Document Corpus
15 curated documents: UN reports, legal texts, human rights reports,
historical agreements, and statistical datasets (1917–2024).
        """)

    with col2:
        st.markdown("### 🛠️ Tech Stack")
        for icon, tech in [
            ("🐍", "Python 3.11+"), ("🎈", "Streamlit 1.x"),
            ("🔍", "FAISS"), ("🤖", "OpenAI SDK"),
            ("📄", "pdfplumber / Tesseract"), ("🗺️", "Leaflet.js"),
            ("☁️", "Hugging Face Spaces"), ("🔑", "AI Grid API"),
        ]:
            st.markdown(
                f'<div class="card" style="padding:.5rem 1rem;margin-bottom:.4rem;">{icon} &nbsp;<b>{tech}</b></div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📊 System Status")
        idx_ok = "✅ Loaded" if (PIPELINE_AVAILABLE and pdf_pipeline and getattr(pdf_pipeline, "index", None) is not None) else "❌ Not loaded"
        emb_ok = "✅ OK" if os.getenv("EMBED_API_KEY") else "❌ Missing"
        llm_ok = "✅ OK" if os.getenv("LLM_API_KEY") else "❌ Missing"
        pip_ok = "✅ Available" if PIPELINE_AVAILABLE else "❌ Not found"
        st.markdown(f"""
        <div class="card">
            <div>📦 pdf_pipeline: <b style="color:#4caf70;">{pip_ok}</b></div>
            <div>🗂️ Index: <b style="color:#4caf70;">{idx_ok}</b></div>
            <div>🔑 Embed Key: <b style="color:#4caf70;">{emb_ok}</b></div>
            <div>🤖 LLM Key: <b style="color:#4caf70;">{llm_ok}</b></div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 11 — VOICE INPUT 🎙️ (+10)
# ═══════════════════════════════════════════════
with tabs[10]:
    st.markdown('## 🎙️ Voice Input <span class="feature-badge">+10 pts</span>', unsafe_allow_html=True)
    st.markdown("Record a voice question and get a text answer. Uses browser Web Speech API.")

    lang = st.radio("Language", ["English", "Arabic"], horizontal=True, key="voice_lang")
    lang_code = "ar-SA" if lang == "Arabic" else "en-US"

    voice_html = f"""
    <div style="background:#161616;border:1px solid #252525;border-radius:10px;padding:1.5rem;font-family:'Source Sans 3',sans-serif;">
        <button id="startBtn" onclick="startRecording()"
            style="background:#1e6e3a;color:#fff;border:none;border-radius:6px;padding:.6rem 1.4rem;
                   font-size:1rem;cursor:pointer;margin-right:10px;font-weight:600;">
            🎙️ Start Recording
        </button>
        <button id="stopBtn" onclick="stopRecording()" disabled
            style="background:#444;color:#fff;border:none;border-radius:6px;padding:.6rem 1.4rem;
                   font-size:1rem;cursor:pointer;font-weight:600;">
            ⏹ Stop
        </button>
        <div id="status" style="margin-top:1rem;color:#4caf70;font-size:.9rem;"></div>
        <div id="transcript" style="margin-top:1rem;background:#0e0e0e;border:1px solid #333;
             border-radius:8px;padding:1rem;color:#e5e5e5;min-height:60px;font-size:1rem;
             direction:{'rtl' if lang == 'Arabic' else 'ltr'};"></div>
        <button onclick="copyTranscript()"
            style="margin-top:.8rem;background:#1a3a26;color:#4caf70;border:1px solid #1e6e3a;
                   border-radius:6px;padding:.4rem 1rem;cursor:pointer;font-size:.85rem;">
            📋 Copy to Clipboard
        </button>
    </div>
    <script>
    let recognition;
    let fullTranscript = '';
    function startRecording() {{
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            document.getElementById('status').innerText = '❌ Speech recognition not supported in this browser. Use Chrome.';
            return;
        }}
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.lang = '{lang_code}';
        recognition.continuous = true;
        recognition.interimResults = true;
        fullTranscript = '';
        recognition.onstart = () => {{
            document.getElementById('status').innerText = '🔴 Recording...';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stopBtn').style.background = '#c0392b';
        }};
        recognition.onresult = (e) => {{
            let interim = '';
            for (let i = e.resultIndex; i < e.results.length; i++) {{
                if (e.results[i].isFinal) fullTranscript += e.results[i][0].transcript + ' ';
                else interim += e.results[i][0].transcript;
            }}
            document.getElementById('transcript').innerText = fullTranscript + interim;
        }};
        recognition.onerror = (e) => {{
            document.getElementById('status').innerText = '❌ Error: ' + e.error;
        }};
        recognition.onend = () => {{
            document.getElementById('status').innerText = '✅ Recording stopped. Copy text above and paste it in Smart Chat.';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('stopBtn').style.background = '#444';
        }};
        recognition.start();
    }}
    function stopRecording() {{ if (recognition) recognition.stop(); }}
    function copyTranscript() {{
        const text = document.getElementById('transcript').innerText;
        navigator.clipboard.writeText(text).then(() => {{
            document.getElementById('status').innerText = '📋 Copied to clipboard!';
        }});
    }}
    </script>
    """
    components.html(voice_html, height=280)
    st.info("💡 After recording, copy the transcript and paste it into the **💬 Smart Chat** tab to get an answer.")

# ═══════════════════════════════════════════════
# TAB 12 — AUTO-TRANSLATE 🌐 (+5)
# ═══════════════════════════════════════════════
with tabs[11]:
    st.markdown('## 🌐 Auto-Translate <span class="feature-badge">+5 pts</span>', unsafe_allow_html=True)
    st.markdown("Translate answers between Arabic and English with one click.")

    tr_col1, tr_col2 = st.columns(2)
    with tr_col1:
        src_lang = st.selectbox("From", ["English", "Arabic", "French"], key="tr_src")
    with tr_col2:
        tgt_lang = st.selectbox("To", ["Arabic", "English", "French"], key="tr_tgt")

    tr_text = st.text_area("Text to translate", height=150,
                           placeholder="Paste any text — answers, summaries, excerpts...", key="tr_text")

    if st.button("🌐 Translate", key="tr_btn"):
        if not tr_text.strip():
            st.warning("Please enter text to translate.")
        else:
            with st.spinner(f"Translating {src_lang} → {tgt_lang}..."):
                result = llm_call(
                    f"Translate the following text from {src_lang} to {tgt_lang}.\n"
                    f"Preserve the meaning, tone, and formatting. Output ONLY the translated text.\n\n"
                    f"Text:\n{tr_text}",
                    system=f"You are a professional translator specializing in {src_lang} and {tgt_lang}, "
                           f"with expertise in political and humanitarian terminology.",
                    max_tokens=1500,
                )
            is_rtl = tgt_lang == "Arabic"
            direction = "rtl" if is_rtl else "ltr"
            st.markdown(
                f'<div class="card" style="direction:{direction};text-align:{"right" if is_rtl else "left"};'
                f'font-size:1.05rem;line-height:1.8;">{result}</div>',
                unsafe_allow_html=True
            )
            st.session_state.translate_history.append({
                "from": src_lang, "to": tgt_lang, "original": tr_text[:100], "result": result
            })

    # Quick-translate last chat answer
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick-Translate Last Chat Answer")
    if st.session_state.chat_messages:
        last_answers = [m for m in st.session_state.chat_messages if m["role"] == "assistant"]
        if last_answers:
            last = last_answers[-1]["content"]
            target = st.radio("Translate to", ["Arabic", "English"], horizontal=True, key="qt_lang")
            if st.button("⚡ Translate Last Answer", key="qt_btn"):
                with st.spinner("Translating..."):
                    translated = llm_call(
                        f"Translate to {target}. Output ONLY the translation:\n\n{last}",
                        max_tokens=1200,
                    )
                is_rtl = target == "Arabic"
                st.markdown(
                    f'<div class="card" style="direction:{"rtl" if is_rtl else "ltr"};">{translated}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No assistant answers in chat history yet.")
    else:
        st.info("Chat history is empty. Ask questions in the Smart Chat tab first.")

# ═══════════════════════════════════════════════
# TAB 13 — ADVANCED ANALYTICS 📊 (+5)
# ═══════════════════════════════════════════════
with tabs[12]:
    st.markdown('## 📊 Advanced Analytics <span class="feature-badge">+5 pts</span>', unsafe_allow_html=True)
    st.markdown("Sentiment trends, entity frequency, and document similarity.")

    an_topic = st.text_input("Analysis topic", placeholder="e.g. violence, displacement, negotiations", key="an_topic")

    an_col1, an_col2, an_col3 = st.columns(3)
    with an_col1:
        do_sentiment = st.checkbox("Sentiment Analysis", value=True, key="an_sent")
    with an_col2:
        do_entities = st.checkbox("Entity Extraction", value=True, key="an_ent")
    with an_col3:
        do_similarity = st.checkbox("Document Similarity", value=False, key="an_sim")

    if st.button("📊 Run Analytics", key="an_btn") and an_topic:
        with st.spinner("Running analysis..."):
            retrieved = safe_retrieve(an_topic, 8)
            context = "\n\n".join(
                f"(Source: {m.get('source','?')} p.{m.get('page','?')}):\n{t}"
                for t, m in retrieved
            ) if retrieved else "No documents retrieved."

            results = {}

            if do_sentiment:
                results["sentiment"] = llm_call(
                    f"Perform sentiment analysis on the following texts about '{an_topic}'.\n"
                    f"For each source, rate sentiment: Positive / Neutral / Negative and explain briefly.\n"
                    f"Then give an overall sentiment trend.\n\nContext:\n{context[:2500]}",
                    system="You are a sentiment analysis expert.",
                    max_tokens=700,
                )

            if do_entities:
                results["entities"] = llm_call(
                    f"Extract named entities from the text about '{an_topic}'.\n"
                    f"Categorize as: PERSON, ORGANIZATION, LOCATION, DATE, EVENT.\n"
                    f"List top 5 per category with frequency estimate.\n\nContext:\n{context[:2500]}",
                    system="You are an expert in named entity recognition.",
                    max_tokens=700,
                )

            if do_similarity:
                docs = get_doc_list()
                if len(docs) >= 2:
                    results["similarity"] = llm_call(
                        f"Based on the context retrieved for '{an_topic}', estimate which documents "
                        f"are most similar in content and theme. Rank by similarity.\n\nContext:\n{context[:2000]}\n\nDocuments: {', '.join(docs[:10])}",
                        max_tokens=500,
                    )

        if "sentiment" in results:
            st.markdown("### 😊 Sentiment Analysis")
            st.markdown(f'<div class="card">{results["sentiment"]}</div>', unsafe_allow_html=True)

        if "entities" in results:
            st.markdown("### 🏷️ Entity Extraction")
            st.markdown(f'<div class="card">{results["entities"]}</div>', unsafe_allow_html=True)

        if "similarity" in results:
            st.markdown("### 🔗 Document Similarity")
            st.markdown(f'<div class="card">{results["similarity"]}</div>', unsafe_allow_html=True)

    elif not an_topic:
        st.info("Enter a topic above and click Run Analytics.")

# ═══════════════════════════════════════════════
# TAB 14 — EXPORT CHAT 💾 (+5)
# ═══════════════════════════════════════════════
with tabs[13]:
    st.markdown('## 💾 Export Chat <span class="feature-badge">+5 pts</span>', unsafe_allow_html=True)
    st.markdown("Export your full chat history as PDF or JSON.")

    if not st.session_state.chat_messages:
        st.info("No chat history yet. Start a conversation in the **💬 Smart Chat** tab.")
    else:
        st.markdown(f"**{len(st.session_state.chat_messages)} messages** in current session.")

        # Preview
        with st.expander("👁️ Preview Chat History"):
            for msg in st.session_state.chat_messages:
                role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg['content'][:300]}{'...' if len(msg['content']) > 300 else ''}")
                st.markdown("---")

        ex_col1, ex_col2 = st.columns(2)

        with ex_col1:
            st.markdown("### 📄 Export as JSON")
            export_data = {
                "exported_at": datetime.datetime.now().isoformat(),
                "total_messages": len(st.session_state.chat_messages),
                "messages": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "sources": m.get("sources", []),
                    }
                    for m in st.session_state.chat_messages
                ],
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"palestine_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="dl_json"
            )

        with ex_col2:
            st.markdown("### 📋 Export as Text")
            lines = [
                "Palestine RAG Chatbot — Chat Export",
                f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 50,
                ""
            ]
            for msg in st.session_state.chat_messages:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                lines.append(f"[{role}]")
                lines.append(msg["content"])
                if msg.get("sources"):
                    lines.append("Sources: " + ", ".join(f"{s['source']} p.{s['page']}" for s in msg["sources"]))
                lines.append("")
            txt = "\n".join(lines)
            st.download_button(
                label="⬇️ Download TXT",
                data=txt,
                file_name=f"palestine_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="dl_txt"
            )

        # HTML export (printable / PDF-ready)
        st.markdown("### 🖨️ Export as Printable HTML (Save as PDF)")
        html_lines = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Palestine RAG Chatbot - Chat Export</title>",
            "<style>body{font-family:Georgia,serif;max-width:800px;margin:2rem auto;color:#222;line-height:1.7;}",
            "h1{color:#1e6e3a;}.user{background:#f0f7f0;border-left:3px solid #1e6e3a;padding:.8rem;margin:1rem 0;border-radius:4px;}",
            ".assistant{background:#f9f9f9;border-left:3px solid #999;padding:.8rem;margin:1rem 0;border-radius:4px;}",
            ".role{font-weight:700;color:#1e6e3a;margin-bottom:.3rem;}.source{font-size:.8rem;color:#666;}</style></head><body>",
            f"<h1>🇵🇸 Palestine RAG Chatbot</h1><p>Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p><hr>",
        ]
        for msg in st.session_state.chat_messages:
            role_label = "You" if msg["role"] == "user" else "Assistant"
            css_class = "user" if msg["role"] == "user" else "assistant"
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f'<div class="{css_class}"><div class="role">{role_label}</div>{content}')
            if msg.get("sources"):
                src_str = " | ".join(f"{s['source']} p.{s['page']}" for s in msg["sources"])
                html_lines.append(f'<div class="source">Sources: {src_str}</div>')
            html_lines.append("</div>")
        html_lines.append("</body></html>")
        html_export = "\n".join(html_lines)
        st.download_button(
            label="⬇️ Download HTML (→ PDF)",
            data=html_export,
            file_name=f"palestine_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key="dl_html"
        )
        st.caption("Open the HTML file in your browser → File → Print → Save as PDF")

# ═══════════════════════════════════════════════
# TAB 15 — MULTI-MODEL COMPARISON 🤖 (+10)
# ═══════════════════════════════════════════════
with tabs[14]:
    st.markdown('## 🤖 Multi-Model Comparison <span class="feature-badge">+10 pts</span>', unsafe_allow_html=True)
    st.markdown("Compare answers from 2 LLMs side-by-side for the same question.")

    available_models = [
        "gpt-oss-120b",
        "gpt-oss-70b",
        "gpt-oss-32b",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet",
    ]

    mm_col1, mm_col2 = st.columns(2)
    with mm_col1:
        model_a = st.selectbox("Model A", available_models, index=0, key="mm_model_a")
    with mm_col2:
        model_b = st.selectbox("Model B", available_models, index=1, key="mm_model_b")

    mm_query = st.text_area("Question to compare", height=100,
                            placeholder="Enter your question...", key="mm_query")
    use_rag = st.checkbox("Use RAG context (retrieve from documents)", value=True, key="mm_rag")

    if st.button("⚡ Compare Models", key="mm_btn"):
        if not mm_query.strip():
            st.warning("Please enter a question.")
        elif model_a == model_b:
            st.warning("Please select two different models.")
        else:
            context = ""
            sources = []
            if use_rag:
                retrieved = safe_retrieve(mm_query, 5)
                if retrieved:
                    context = "\n\n".join(
                        f"(Source: {m.get('source','?')} p.{m.get('page','?')}):\n{t}"
                        for t, m in retrieved
                    )
                    sources = [{"source": m.get("source", "?"), "page": m.get("page", "?")} for _, m in retrieved]

            prompt = (
                f"Context:\n{context}\n\nQuestion:\n{mm_query}"
                if context else mm_query
            )
            system = "Answer using the context provided. Cite sources." if context else None

            with st.spinner("Querying both models..."):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fut_a = executor.submit(llm_call, prompt, system, 1000, model_a)
                    fut_b = executor.submit(llm_call, prompt, system, 1000, model_b)
                    ans_a = fut_a.result()
                    ans_b = fut_b.result()

            st.markdown("---")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"### 🅰️ {model_a}")
                st.markdown(f'<div class="card" style="min-height:200px;">{ans_a}</div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f"### 🅱️ {model_b}")
                st.markdown(f'<div class="card" style="min-height:200px;">{ans_b}</div>', unsafe_allow_html=True)

            if sources:
                st.markdown("**Shared RAG Sources:**")
                chips = " ".join(
                    f'<span class="source-chip">📄 {s["source"]} p.{s["page"]}</span>' for s in sources
                )
                st.markdown(chips, unsafe_allow_html=True)

            # Auto-comparison
            st.markdown("### 🔍 Auto-Comparison Analysis")
            with st.spinner("Analyzing differences..."):
                comparison = llm_call(
                    f"Compare these two answers to the question: '{mm_query}'\n\n"
                    f"Answer A ({model_a}):\n{ans_a}\n\n"
                    f"Answer B ({model_b}):\n{ans_b}\n\n"
                    f"Analyze: 1) Key differences 2) Which is more accurate/comprehensive 3) Which cites sources better 4) Overall recommendation.",
                    max_tokens=600,
                )
            st.markdown(f'<div class="card">{comparison}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#444;font-size:.75rem;">'
    '🇵🇸 Palestine RAG Chatbot · Streamlit · FAISS · GPT-OSS · For academic use only'
    '</p>',
    unsafe_allow_html=True
)
