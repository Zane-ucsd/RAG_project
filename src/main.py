from retriever import retrieve_top_k
from generator import generate_answer

def main():
    print("🚀 RAG 问答系统启动！")
    query = input("请输入你的问题: ")
    
    # 1. 检索相关文档
    retrieved_docs = retrieve_top_k(query)
    print(f"🔍 相关文档: {retrieved_docs}")

    # 2. 生成答案
    response = generate_answer(query, "\n".join(retrieved_docs))
    print(f"🤖 AI 回答: {response}")

if __name__ == "__main__":
    main()
    