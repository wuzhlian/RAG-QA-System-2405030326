# RAG智能问答系统

基于本地知识库的检索增强生成（RAG）智能问答系统，利用Ollama本地大模型和LangChain框架构建。

## 功能特点

- 📁 支持上传PDF、DOCX、TXT格式的文档
- 📚 自动构建本地知识库
- 🔍 基于向量检索的智能问答
- 💬 支持多轮对话交互
- 📊 实时显示知识库状态

## 环境要求

- Python 3.9+
- Ollama（已安装deepseek-r1:7b或qwen2:7b模型）

## 安装步骤

1. **安装Ollama**
   ```bash
   # 下载并安装Ollama
   # 官方网站: https://ollama.com/
   ```

2. **下载大语言模型**
   ```bash
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text
   ```

3. **创建虚拟环境**
   ```bash
   python -m venv venv
   ```

4. **激活虚拟环境**
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

5. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

1. **启动Ollama服务**
   ```bash
   ollama serve
   ```

2. **运行Web应用**
   ```bash
   streamlit run app.py
   ```

3. **使用步骤**
   - 在左侧面板上传文档（支持PDF、DOCX、TXT格式）
   - 点击"构建/更新知识库"按钮
   - 在右侧面板输入问题并点击"提问"
   - 查看回答结果

## 关键技术点

### RAG流程
1. **文档加载**：支持PDF、DOCX、TXT等多种格式
2. **文本分块**：使用RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化**：使用Ollama内置的nomic-embed-text模型
4. **向量存储**：使用Chroma向量数据库
5. **检索**：基于相似度检索，返回最相关的3个文本块
6. **生成**：使用deepseek-r1:7b模型生成回答

### 系统提示词设计
- 要求模型基于提供的参考文档回答
- 如果文档中没有相关信息，明确返回"文档中未找到相关答案"
- 禁止编造信息

## 项目结构

```
RAG-QA-System/
├── app.py              # Streamlit Web应用主文件
├── rag_utils.py        # RAG工具函数（文档加载、向量化、检索）
├── rag_chain.py        # RAG问答链实现
├── test_ollama.py      # Ollama API测试脚本
├── requirements.txt    # 依赖列表
├── .gitignore          # Git忽略配置
├── documents/          # 示例文档目录
│   ├── nlp_introduction.txt
│   ├── transformer_model.txt
│   ├── bert_model.txt
│   ├── word_embedding.txt
│   └── nlp_applications.txt
└── chroma_db/          # 向量数据库存储目录
```

## 测试问答示例

### 相关问题（文档中有答案）
1. 什么是自然语言处理？
2. Transformer模型的核心组件有哪些？
3. 什么是词嵌入？
4. BERT模型的特点是什么？
5. NLP有哪些应用场景？

### 无关问题（文档中无答案）
1. 量子计算的原理是什么？
2. 人工智能的发展历史

## 已知问题与改进方向

- 当前仅支持单轮对话，后续可优化为多轮对话
- 可添加更多文档格式支持（如Markdown、HTML等）
- 可考虑添加文档预览功能
- 可优化向量检索算法，提高检索准确度

## 许可证

MIT License
