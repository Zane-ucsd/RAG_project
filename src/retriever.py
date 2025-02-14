import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, "vector.index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.npy")

# 加载嵌入模型
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_top_k(query, k=2):
    """ 检索与查询最相关的 K 个文档 """
    # 加载索引和文档
    index = faiss.read_index(VECTOR_INDEX_PATH)
    documents = np.load(DOCS_PATH, allow_pickle=True)

    # 计算查询的向量
    query_embedding = embed_model.encode([query], convert_to_numpy=True)

    # 在 FAISS 中搜索最相关的 K 个文档
    _, retrieved_indices = index.search(query_embedding, k)
    retrieved_docs = [documents[idx] for idx in retrieved_indices[0]]

    return retrieved_docs

if __name__ == "__main__":
    query = "小青马是人还是东西？"
    retrieved_docs = retrieve_top_k(query)
    print(f"🔍 检索到的相关文档: {retrieved_docs}")
