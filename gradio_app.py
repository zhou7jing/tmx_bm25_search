import gradio as gr

# ✅ 导入你现有的代码（注意文件名）
from main import (
    build_or_load_index_from_tmx_dir,
    retrieve,
    detect_query_lang,
    print_hits,
    RAGAgent,
    HelloAgentsLLM
)

# 全局缓存 index（避免每次重建）
GLOBAL_INDEX = None


# =========================
# 1️⃣ 初始化索引
# =========================
def load_index_fn(tmx_dir, work_dir, force_rebuild):
    global GLOBAL_INDEX

    try:
        index = build_or_load_index_from_tmx_dir(
            tmx_dir=tmx_dir,
            work_dir=work_dir,
            force_rebuild=force_rebuild
        )
        GLOBAL_INDEX = index
        return f"✅ 索引加载成功，items={len(index.items)}"
    except Exception as e:
        return f"❌ 索引加载失败: {e}"


# =========================
# 2️⃣ Search模式
# =========================
def search_fn(query, top_k):
    global GLOBAL_INDEX

    if GLOBAL_INDEX is None:
        return "⚠️ 请先加载索引"

    lang = detect_query_lang(query)
    hits = retrieve(GLOBAL_INDEX, query, top_k=top_k, lang_hint=lang)

    if not hits:
        return "（无匹配）"

    output = []
    seen = set()

    for it, score in hits:
        u = it.aligned
        m = u.metadata
        uid = (m.get("source_file"), m.get("tu_index"))

        if uid in seen:
            continue
        seen.add(uid)

        output.append(
            f"score={score:.4f}\n"
            f"CN: {u.zh_text}\n"
            f"EN: {u.en_text}\n"
            f"---"
        )

        if len(output) >= 8:
            break

    return "\n\n".join(output)


# =========================
# 3️⃣ RAG模式
# =========================
def rag_fn(query, top_k):
    global GLOBAL_INDEX

    if GLOBAL_INDEX is None:
        return "⚠️ 请先加载索引"

    try:
        llm = HelloAgentsLLM()
        agent = RAGAgent(llm, GLOBAL_INDEX)
        return agent.ask(query, top_k=top_k)
    except Exception as e:
        return f"❌ RAG 失败: {e}"

#上传tmx按钮
import shutil
import os

def upload_tmx_fn(file, tmx_dir):
    if file is None:
        return "⚠️ 未选择文件"

    try:
        os.makedirs(tmx_dir, exist_ok=True)

        filename = os.path.basename(file.name)
        save_path = os.path.join(tmx_dir, filename)

        # ✅ 防止覆盖（可选）
        if os.path.exists(save_path):
            return f"⚠️ 文件已存在: {filename}"

        shutil.copy(file.name, save_path)

        return f"✅ 上传成功: {filename}，请点击“加载索引”刷新"

    except Exception as e:
        return f"❌ 上传失败: {e}"

# =========================
# 4️⃣ Gradio UI
# =========================
with gr.Blocks(title="TMX RAG 系统") as demo:

    gr.Markdown("## 🧠 TMX 双语检索 / RAG 系统")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 配置")

            tmx_dir = gr.Textbox(label="TMX目录", value="./tmx")
            work_dir = gr.Textbox(label="工作目录", value="./bm25_work")
            rebuild = gr.Checkbox(label="强制重建索引", value=False)

            load_btn = gr.Button("🚀 加载/构建索引")
            status_output = gr.Textbox(label="状态")

            gr.Markdown("### 📤 上传 TMX 文件")

            upload_file = gr.File(label="选择 TMX 文件", file_types=[".tmx"])
            upload_btn = gr.Button("上传")
            upload_output = gr.Textbox(label="上传状态")


        with gr.Column(scale=2):
            gr.Markdown("### 🔍 查询")

            query = gr.Textbox(label="输入Query")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")

            with gr.Row():
                search_btn = gr.Button("BM25检索")
                rag_btn = gr.Button("RAG回答")

            result_output = gr.Textbox(
                label="输出",
                lines=20
            )

    # ✅ 绑定事件
    load_btn.click(
        load_index_fn,
        inputs=[tmx_dir, work_dir, rebuild],
        outputs=status_output
    )

    upload_btn.click(
        upload_tmx_fn,
        inputs=[upload_file, tmx_dir],
        outputs=upload_output
    )

    search_btn.click(
        search_fn,
        inputs=[query, top_k],
        outputs=result_output
    )

    rag_btn.click(
        rag_fn,
        inputs=[query, top_k],
        outputs=result_output
    )


# =========================
# 启动
# =========================
if __name__ == "__main__":
    demo.launch(server_port=7777)