import os
import numpy as np
import faiss
import pickle

from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# =====================
# GLOBAL STATE
# =====================
texts = []
metadatas = []
index = None

DB_PATH = "db/faiss.index"
TEXT_PATH = "db/texts.pkl"
META_PATH = "db/meta.pkl"

# =====================
# API CLIENT
# =====================
print("🔌 [INIT] Embedding client...")

embed_client = OpenAI(
    api_key=os.getenv("EMBED_API_KEY"),
    base_url="http://app.ai-grid.io:4000/v1"
)

# =====================
# TEXT SPLITTER
# =====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,  # 🔥 better than 400
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

# =====================
# EMBEDDING (BATCHED ⚡)
# =====================
def embed_texts(texts_list, batch_size=64):
    print(f"⚡ [EMBED] Total texts: {len(texts_list)}")

    vectors = []

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i + batch_size]

        try:
            res = embed_client.embeddings.create(
                model="Alibaba-NLP/gte-Qwen2-7B-instruct",
                input=batch
            )

            batch_vectors = [d.embedding for d in res.data]
            vectors.extend(batch_vectors)

            print(f"⚡ Embedded {i + len(batch)}/{len(texts_list)}")

        except Exception as e:
            print("❌ [EMBED ERROR]:", e)

    return vectors


def embed_query(query):
    print("🔎 [EMBED QUERY]:", query)

    res = embed_client.embeddings.create(
        model="Alibaba-NLP/gte-Qwen2-7B-instruct",
        input=query
    )
    return res.data[0].embedding

# =====================
# OCR
# =====================

# =====================
# PROCESS PDF
# =====================
def process_pdf(file_path):
    print(f"📄 [PROCESS] {file_path}")

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    full_text = " ".join([d.page_content.strip() for d in docs])

    if len(full_text) < 50:
        print("⚠️ Using OCR...")
        docs = ocr_pdf(file_path)
    else:
        for d in docs:
            d.metadata["source"] = os.path.basename(file_path)

    chunks = splitter.split_documents(docs)

    print(f"📦 [CHUNKS] {len(chunks)} from {os.path.basename(file_path)}")

    return chunks

# =====================
# BUILD INDEX
# =====================
def build_index(data_path="data"):
    global texts, metadatas, index

    all_chunks = []

    print("🚀 [BUILD] Starting indexing...")

    for file in os.listdir(data_path):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(data_path, file)

        try:
            chunks = process_pdf(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Error: {file} -> {e}")

    print(f"\n📦 TOTAL CHUNKS: {len(all_chunks)}\n")

    # Debug sample
    for i in range(min(3, len(all_chunks))):
        print(f"\n--- SAMPLE CHUNK {i} ---")
        print(all_chunks[i].page_content[:200])
        print(all_chunks[i].metadata)

    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]

    embeddings = np.array(embed_texts(texts)).astype("float32")

    print("🧠 [FAISS] Normalizing vectors...")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(embeddings)

    os.makedirs("db", exist_ok=True)
    faiss.write_index(index, DB_PATH)

    with open(TEXT_PATH, "wb") as f:
        pickle.dump(texts, f)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadatas, f)

    print("✅ Index built and saved")

# =====================
# LOAD INDEX
# =====================
def load_index():
    global texts, metadatas, index

    if not os.path.exists(DB_PATH):
        return False

    print("⚡ Loading existing index...")

    index = faiss.read_index(DB_PATH)

    with open(TEXT_PATH, "rb") as f:
        texts = pickle.load(f)

    with open(META_PATH, "rb") as f:
        metadatas = pickle.load(f)

    return True

# =====================
# INIT
# =====================
def initialize(data_path="data"):
    if not load_index():
        print("🚀 Building index...")
        build_index(data_path)
    else:
        print("✅ Using cached index")

# =====================
# RETRIEVE
# =====================
def retrieve(query, k=5):
    print("\n🔎 [RETRIEVE] Query:", query)

    q_emb = embed_query(query)
    q_emb = np.array([q_emb]).astype("float32")

    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k)

    results = []

    print("\n📊 [RETRIEVE RESULTS]\n")

    for score, i in zip(D[0], I[0]):
        print(f"Score: {score:.4f}")
        print("Source:", metadatas[i])
        print("Preview:", texts[i][:150])
        print("-----")

        results.append((texts[i], metadatas[i]))

    return results
