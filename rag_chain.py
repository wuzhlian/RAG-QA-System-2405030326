from typing import Optional

from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from rag_utils import get_vector_db, search_similar

LLM_MODEL = "deepseek-r1:7b"

def create_rag_chain(vector_db=None):
    """创建RAG问答链"""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    
    if vector_db is None:
        vector_db = get_vector_db()
        if vector_db is None:
            return None
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    template = """
    你是一个基于知识库的智能问答助手。请根据提供的参考文档回答用户的问题。
    
    参考文档：
    {context}
    
    对话历史：
    {chat_history}
    
    用户问题：
    {question}
    
    请严格按照以下规则回答：
    1. 必须基于提供的参考文档内容进行回答
    2. 如果文档中没有相关信息，请明确说"文档中未找到相关答案"
    3. 回答要简洁明了，直接回应用户的问题
    4. 不要编造信息，不要添加文档中没有的内容
    
    回答：
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain

def query_rag(chain, question: str) -> dict:
    """执行RAG查询"""
    try:
        result = chain({"question": question})
        return {
            "answer": result.get("answer", "").strip(),
            "sources": [doc.page_content[:200] for doc in result.get("source_documents", [])]
        }
    except Exception as e:
        print(f"查询失败: {e}")
        return {"answer": "文档中未找到相关答案", "sources": []}

def query_without_chain(question: str) -> str:
    """直接使用LLM回答（不使用RAG）"""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    response = llm.invoke(question)
    return response.content

if __name__ == "__main__":
    print("=== RAG问答测试 ===")
    
    vector_db = get_vector_db()
    if vector_db is None:
        print("警告：未找到向量数据库，请先构建知识库")
        exit()
    
    chain = create_rag_chain(vector_db)
    if chain is None:
        print("无法创建RAG链")
        exit()
    
    test_questions = [
        "什么是自然语言处理？",
        "Transformer模型的核心组件有哪些？",
        "什么是词嵌入？",
        "BERT模型的特点是什么？",
        "NLP有哪些应用场景？",
        "量子计算的原理是什么？",
        "人工智能的发展历史"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = query_rag(chain, question)
        print(f"回答: {result['answer']}")
        if result['sources']:
            print("参考来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source}...")
