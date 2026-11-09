# LangChain + Codex RAG

这是一个可本地运行的检索增强生成系统：LangChain 负责文档加载、切分、向量检索和提示词编排，最终回答由本机 Codex CLI 生成。

Codex 的模型、Responses API 地址和认证信息直接继承自 ~/.codex/config.toml 与 ~/.codex/auth.json。项目不会复制、解析或保存其中的密钥。

## 架构

    文档目录
       -> LangChain loaders
       -> RecursiveCharacterTextSplitter
       -> 本地嵌入（默认离线字符 n-gram；可选 FastEmbed/ONNX）
       -> FAISS 本地索引
       -> Top-K 检索
       -> LangChain ChatPromptTemplate
       -> codex exec（临时会话、只读沙箱）
       -> 带来源编号的回答

## 安装

推荐使用 Python 3.11 或 3.12。项目也会在支持依赖轮子的更新版本 Python 上运行。

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e ".[dev]"

确认 Codex 已登录，并且配置位于 ~/.codex：

    codex --version
    codex-rag doctor

本项目只调用 Codex CLI，不需要在 .env 中再次填写 API key。

## 使用

1. 将 TXT、Markdown、RST、Python、JSON、YAML、CSV、HTML、PDF 或 DOCX 文件放入 knowledge 目录。

2. 创建索引：

    codex-rag ingest

也可以指定其他目录：

    codex-rag ingest /path/to/documents

3. 提问：

    codex-rag ask "这个知识库主要讲了什么？"

4. 交互模式：

    codex-rag chat

也可以不安装命令行入口，直接运行：

    PYTHONPATH=src python -m codex_rag doctor

## 配置

复制 .env.example 为 .env 后可覆盖默认值：

- RAG_DOCS_DIR：知识库目录，默认 knowledge
- RAG_INDEX_DIR：FAISS 索引目录，默认 .rag_index
- RAG_EMBEDDING_MODEL：默认 local-hash-v1，完全离线且对中文友好
- RAG_EMBEDDING_CACHE_DIR：ONNX 模型缓存目录，默认 .cache/fastembed
- RAG_TOP_K：每次检索片段数，默认 4
- RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP：文档切分参数
- CODEX_HOME：Codex 配置目录，默认 ~/.codex
- CODEX_MODEL：留空时继承 Codex config.toml 中的模型
- CODEX_REASONING_EFFORT：Codex 推理强度，默认 medium
- CODEX_TIMEOUT_SECONDS：单次回答超时，默认 180 秒

默认嵌入器采用字符 n-gram 特征哈希，不下载模型，适合离线运行。它属于词法向量检索；如果环境可以访问 Hugging Face，可启用更强的多语言语义嵌入：

    python -m pip install -e ".[semantic]"
    RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 codex-rag ingest

FastEmbed 使用 ONNX，不需要 PyTorch 或 CUDA。

## Python API

    from codex_rag import RAGService, Settings

    settings = Settings.from_env()
    rag = RAGService(settings)
    rag.ingest()
    result = rag.answer("系统如何复用 Codex 配置？")
    print(result.text)
    print(result.sources)

## 安全边界

- Codex 调用使用 ephemeral 会话，不写入 Codex 对话历史。
- Codex 在 read-only 沙箱中运行，并禁用项目规则与网络检索。
- 知识片段通过标准输入发送给 Codex，不写入 ~/.codex。
- FAISS 向量与文档元数据分别保存为原生索引和 JSON，不使用 pickle 反序列化。
