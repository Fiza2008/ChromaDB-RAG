# 🔍 RAG + ChromaDB (Local Vector Database) + OpenAI

Minimal RAG implementation using ChromaDB as the vector database.

## ✨ Features
- Store embeddings locally using ChromaDB
- Query documents using similarity search
- No cloud dependencies, 100% local vector DB

## ⚙️ Tech Stack
| Component | Tool |
|----------|------|
| Vector DB | ChromaDB |
| LLM | OpenAI |
| Embeddings | OpenAI embeddings |
| Language Framework | LangChain |

---

## 🚀 Run Code

```bash
pip install -r requirements.txt
python main.py


---

### ✅ Code Boilerplate

📁 Structure:

chromadb-rag/
│── main.py
│── requirements.txt
│── README.md
│── .env


**main.py**

```python
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db"))
collection = client.create_collection(name="rag-store")

texts = [
    "Western Digital manufactures HDD and SSD storage solutions.",
    "RAG improves LLM accuracy by retrieving external information."
]

embeddings = OpenAIEmbeddings()

for idx, text in enumerate(texts):
    collection.add(
        ids=[str(idx)],
        documents=[text],
        embeddings=[embeddings.embed_query(text)]
    )

query = "What does Western Digital do?"
results = collection.query(query_texts=[query], n_results=1)

llm = ChatOpenAI(model="gpt-4.1-mini")
response = llm.predict(f"Answer based on context: {results['documents']}")
print(response)
