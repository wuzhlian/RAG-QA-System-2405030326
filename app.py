import os
import tempfile
import streamlit as st

from rag_utils import load_document, split_documents, create_vector_db, get_vector_db, get_db_stats, clear_vector_db, check_ollama_connection, check_embedding_model
from rag_chain import create_rag_chain, query_rag

st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="📚",
    layout="wide"
)

def init_session_state():
    """初始化会话状态"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "db_stats" not in st.session_state:
        st.session_state.db_stats = {"documents": 0, "chunks": 0}

def main():
    init_session_state()
    
    st.title("📚 基于本地知识库的RAG智能问答系统")
    st.markdown("---")
    
    ollama_connected = check_ollama_connection()
    model_ready = check_embedding_model()
    
    if not ollama_connected:
        st.error("""
        ❌ **Ollama服务未启动**
        
        请按照以下步骤操作：
        1. 安装Ollama：https://ollama.com/
        2. 启动Ollama服务：`ollama serve`
        3. 下载模型：`ollama pull nomic-embed-text` 和 `ollama pull deepseek-r1:7b`
        
        服务启动后，请刷新页面。
        """)
        return
    
    if not model_ready:
        st.warning("""
        ⚠️ **嵌入模型未下载**
        
        请运行以下命令下载模型：
        ```bash
        ollama pull nomic-embed-text
        ollama pull deepseek-r1:7b
        ```
        
        下载完成后，请刷新页面。
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 文档管理")
        
        uploaded_files = st.file_uploader(
            "上传PDF/DOCX/TXT文档",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        if st.button("🚀 构建/更新知识库", type="primary"):
            if not uploaded_files:
                st.warning("请先上传文档")
            elif not model_ready:
                st.error("请先下载嵌入模型（nomic-embed-text）")
            else:
                with st.spinner("正在处理文档..."):
                    all_documents = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        docs = load_document(tmp_file_path)
                        all_documents.extend(docs)
                        os.unlink(tmp_file_path)
                    
                    if not all_documents:
                        st.error("未能加载任何文档")
                        return
                    
                    chunks = split_documents(all_documents)
                    clear_vector_db()
                    st.session_state.vector_db = create_vector_db(chunks)
                    
                    if st.session_state.vector_db:
                        st.session_state.rag_chain = create_rag_chain(st.session_state.vector_db)
                        st.session_state.db_stats = get_db_stats(st.session_state.vector_db)
                        st.success(f"✅ 知识库构建成功！\n- 文档数: {len(uploaded_files)}\n- 文本块数: {len(chunks)}")
                    else:
                        st.error("知识库构建失败，请检查Ollama服务是否正常运行")
        
        st.markdown("---")
        st.subheader("📊 知识库状态")
        
        db = get_vector_db() if not st.session_state.vector_db else st.session_state.vector_db
        stats = get_db_stats(db)
        
        st.info(f"""
        **当前知识库状态：**
        - 文档数量：{stats['documents']}
        - 文本块数量：{stats['chunks']}
        """)
        
        if st.button("🗑️ 清空知识库"):
            clear_vector_db()
            st.session_state.vector_db = None
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.db_stats = {"documents": 0, "chunks": 0}
            st.success("知识库已清空")
            st.rerun()
    
    with col2:
        st.subheader("💬 问答交互")
        
        if st.session_state.rag_chain is None:
            st.info("请先上传文档并构建知识库")
        else:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(f"**用户**: {question}")
                with st.chat_message("assistant"):
                    st.markdown(f"**助手**: {answer}")
            
            user_input = st.text_input("请输入您的问题：", key="question_input")
            
            if st.button("提问", type="secondary") and user_input.strip():
                with st.spinner("正在思考..."):
                    result = query_rag(st.session_state.rag_chain, user_input.strip())
                    answer = result["answer"]
                    
                    st.session_state.chat_history.append((user_input.strip(), answer))
                    
                    with st.chat_message("user"):
                        st.markdown(f"**用户**: {user_input}")
                    with st.chat_message("assistant"):
                        st.markdown(f"**助手**: {answer}")
                
                st.rerun()
        
        if st.session_state.chat_history and st.session_state.rag_chain:
            if st.button("🗑️ 清空对话历史"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
