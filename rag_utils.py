import os
import json
import requests
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_HOST = "http://localhost:11434"

def check_ollama_connection() -> bool:
    """检查Ollama服务是否可用"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def check_embedding_model() -> bool:
    """检查嵌入模型是否已下载"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json().get("models", [])
            model_names = [model["name"] for model in tags]
            return any(EMBEDDING_MODEL in name for name in model_names)
        return False
    except Exception:
        return False

def load_document(file_path: str) -> List[Document]:
    """加载单个文档并提取文本"""
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"不支持的文件格式: {ext}")
            return []
        
        documents = loader.load()
        print(f"成功加载文档: {file_path}, 页数/段落数: {len(documents)}")
        return documents
    except Exception as e:
        print(f"加载文档失败 {file_path}: {e}")
        return []

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """批量加载文件夹中的所有文档"""
    all_documents = []
    
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return all_documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_documents.extend(docs)
    
    print(f"共加载 {len(all_documents)} 个文档片段")
    return all_documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """将文档分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"文档分块完成，共生成 {len(chunks)} 个文本块")
    return chunks

def create_vector_db(documents: List[Document], persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """创建或更新向量数据库"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"向量数据库创建成功，共存储 {len(documents)} 个文本块")
        return vector_db
    except Exception as e:
        print(f"创建向量数据库失败: {e}")
        return None

def get_vector_db(persist_directory: str = CHROMA_DB_PATH) -> Optional[Chroma]:
    """获取已存在的向量数据库"""
    if not os.path.exists(persist_directory):
        return None
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vector_db
    except Exception as e:
        print(f"加载向量数据库失败: {e}")
        return None

def search_similar(query: str, vector_db: Chroma, k: int = 3) -> List[Document]:
    """搜索相似文本块"""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    return results

def get_db_stats(vector_db: Chroma) -> dict:
    """获取向量数据库统计信息"""
    if vector_db is None:
        return {"documents": 0, "chunks": 0}
    
    try:
        collection = vector_db._collection
        count = collection.count()
        return {"documents": 1, "chunks": count}
    except Exception as e:
        print(f"获取数据库统计失败: {e}")
        return {"documents": 0, "chunks": 0}

def clear_vector_db(persist_directory: str = CHROMA_DB_PATH):
    """清空向量数据库"""
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("向量数据库已清空")
