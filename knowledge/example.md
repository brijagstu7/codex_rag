# 示例知识库

Codex RAG 是一个本地检索增强生成示例。它使用 LangChain 加载与切分文档，使用本地多语言嵌入模型生成向量，并把向量保存在 FAISS 索引中。

回答问题时，系统先从 FAISS 中检索最相关的文档片段，再把这些片段连同问题交给本机 Codex CLI。Codex CLI 从用户的 ~/.codex 目录继承模型、API 地址和认证信息，因此项目本身不保存 API 密钥。

系统支持 TXT、Markdown、RST、Python、JSON、YAML、CSV、HTML、PDF 和 DOCX 文件。

