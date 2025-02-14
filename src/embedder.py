from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 加载嵌入模型
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_documents(documents):
    """ 将文本转换为向量 """
    return embed_model.encode(documents, convert_to_numpy=True)

def build_faiss_index(documents, index_path="data/vector.index", doc_path="data/docs.npy"):
    """ 构建 FAISS 向量索引并保存 """
    embeddings = encode_documents(documents)
    dimension = embeddings.shape[1]

    # 创建 FAISS L2 索引
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 保存索引 & 文档
    faiss.write_index(index, index_path)
    np.save(doc_path, documents)

if __name__ == "__main__":
    docs = [
        "人工智能是计算机科学的一个分支，旨在创建能够执行需要人类智能的任务的系统。",
        "深度学习是一种机器学习方法，使用神经网络来学习数据中的模式。",
        "FAISS 是一个高效的向量搜索库，适用于大规模检索任务。",
        "GPT-4 是 OpenAI 开发的先进语言模型，能够理解和生成自然语言文本。",
        "周子涵十分热爱拳击，但是他的妈妈并不支持他去学习拳击，因为他的妈妈认为拳击是一项危险的运动。",
        "小青马是周子涵的爸爸送给他的礼物，小青马是一匹非常可爱的小马，周子涵非常喜欢它。",
        "杨政熹是一个非常喜欢小动物的人，他经常会给小青马喂食，并且带它去散步。",
        "杨政熹，周子涵，小青马都是ucsd 计算机专业的学生。",
    ]
    build_faiss_index(docs)
