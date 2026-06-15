## 📌 项目简介

本系统用于批量解析 TMX 翻译记忆库文件，构建本地 BM25 索引，并提供基于 Gradio 的可视化检索界面，实现中英双语句对的高效查询与复用。

***

## 🚀 使用流程

### 1️⃣ 准备数据

将所有 `.tmx` 文件放入同一个目录，例如：

```
./tmx_data/
```

***

### 2️⃣ 构建索引

启动程序后点击“构建索引”，系统会自动完成：

* TMX 解析（中英对齐）
* 句对去重
* BM25 检索索引构建
* 缓存写入（提升后续加载速度）

***

### 3️⃣ 进行检索

构建完成后，可以在前端输入：

* 中文 → 查英文
* 英文 → 查中文
* 关键词 → 查相关句对 / 项目内容

***

## ▶️ 运行方式

```bash
python gradio_app.py
```

***

## 🌐 服务地址

* 默认端口：`7778`
* 访问地址（本机）：

```bash
http://localhost:7778
```

如需外网访问，可绑定：

```bash
0.0.0.0:7778
```

***

## ⚙️ 功能特性

* ✅ TMX 批量解析
* ✅ 中英句对自动对齐
* ✅ 项目信息（Project Name）保留
* ✅ 句对去重（提升检索质量）
* ✅ BM25 高效文本检索
* ✅ 本地缓存（避免重复解析）
* ✅ Gradio 可视化界面

***

## 📂 输出与缓存

系统会自动生成：

```
work_dir/
├── cache/            # TMX解析缓存（jsonl）
├── bm25_index.pkl    # 检索索引
├── manifest.json     # 文件索引信息
```

***

## ⚠️ 注意事项

* 首次运行建议开启完整索引构建
* 若 TMX 文件有更新，请重新构建索引
* 大规模 TMX 建议定期去重优化

***

***

# ✅ Optimized README (English Version)

## 📌 Overview

This project processes TMX (Translation Memory eXchange) files, builds a BM25-based search index, and provides a Gradio-powered UI for efficient bilingual sentence retrieval.

***

## 🚀 Workflow

### 1️⃣ Prepare Data

Place all `.tmx` files into a single directory, for example:

```
./tmx_data/
```

***

### 2️⃣ Build Index

Click **“Build Index”** in the UI. The system will:

* Parse TMX files (sentence alignment)
* Deduplicate bilingual pairs
* Build BM25 search index
* Cache parsed results for faster reload

***

### 3️⃣ Search

After indexing, you can:

* Input Chinese → retrieve English
* Input English → retrieve Chinese
* Input keywords → retrieve relevant sentence pairs or project context

***

## ▶️ Run the Application

```bash
python gradio_app.py
```

***

## 🌐 Access

* Default port: `7778`
* Local access:

```bash
http://localhost:7778
```

To enable external access:

```bash
0.0.0.0:7778
```

***

## ⚙️ Features

* ✅ Batch TMX parsing
* ✅ Chinese-English sentence alignment
* ✅ Project metadata extraction (Project Name)
* ✅ Deduplication for better retrieval quality
* ✅ BM25-based search engine
* ✅ Cached intermediate results
* ✅ Gradio web interface

***

## 📂 Outputs

The system generates:

```
work_dir/
├── cache/            # Parsed TMX cache (JSONL)
├── bm25_index.pkl    # BM25 index
├── manifest.json     # File tracking metadata
```

***

## ⚠️ Notes

* First run requires building the index
* Rebuild index if TMX files change
* Deduplication is recommended for large datasets

***

## ✅ 小优化建议（结合你的项目）

你这个 README 还有一个可以再提升的点：

👉 加一段“典型应用场景”，比如：

* 技术手册翻译复用
* 术语一致性查询
* 客户交付 QA

如果你需要，我可以帮你写一版\*\*“GitHub 展示级 README（带架构图 + 示例截图区域）”\*\*。
