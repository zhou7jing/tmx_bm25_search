import os
import re
import json
import time
import pickle
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from openai import OpenAI


# =========================
# 0) LLM 客户端
# =========================
class HelloAgentsLLM:
    """
    兼容 OpenAI 接口的 LLM 客户端，默认使用流式响应。
    环境变量：
      - LLM_MODEL_ID
      - LLM_API_KEY
      - LLM_BASE_URL
      - LLM_TIMEOUT
    """

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "模型ID、API密钥和服务地址必须提供，"
                "或在环境变量中定义（LLM_MODEL_ID/LLM_API_KEY/LLM_BASE_URL）。"
            )

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("✅ 大语言模型响应成功:")
            collected = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected.append(content)
            print()
            return "".join(collected)

        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            return None


# =========================
# 1) 数据结构
# =========================
@dataclass
class AlignedSentence:
    """
    一个对齐单元：一个 TMX TU 对应一个中英双语单元
    """
    pair_key: str
    zh_text: str
    en_text: str
    project_name: str
    metadata: Dict[str, Any]


@dataclass
class IndexItem:
    """
    BM25 的最小索引单元
    一条语言一条 item
    """
    index_text: str
    lang: str                  # "zh" 或 "en"
    aligned: AlignedSentence   # 指回对齐单元
    metadata: Dict[str, Any]


@dataclass
class BM25Index:
    items: List[IndexItem]
    corpus_tokens: List[List[str]]
    bm25: BM25Okapi


# =========================
# 2) 分词：jieba 优先，自动降级
# =========================
def _try_import_jieba():
    try:
        import jieba  # type: ignore
        return jieba
    except Exception:
        return None


_JIEBA = _try_import_jieba()


def tokenize(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []

    if _JIEBA is not None:
        lcut = getattr(_JIEBA, "lcut", None)
        if callable(lcut):
            return [t.strip() for t in lcut(text) if t.strip()]

        cut = getattr(_JIEBA, "cut", None)
        if callable(cut):
            return [t.strip() for t in cut(text) if t.strip()]

    # fallback：中文逐字 + 英文数字串
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


# =========================
# 3) 文本清洗
# =========================
def normalize_sentence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# 4) 工具函数：文件与缓存
# =========================

def discover_tmx_files(tmx_dir: str) -> List[str]:
    paths = []
    for base, _, files in os.walk(tmx_dir):
        for fn in files:
            if fn.lower().endswith(".tmx"):
                paths.append(os.path.join(base, fn))
    return sorted(paths)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def file_digest(path: str) -> str:
    """
    基于文件内容计算 digest，比 size+mtime 更稳
    """
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# =========================
# 5) TMX 解析 -> 对齐单元
# =========================
def parse_tmx_to_aligned_units(
    tmx_path: str,
    cn_langs=("zh-CN", "zh", "zh-Hans", "zh-CHS", "zh_CN"),
    en_langs=("en-US", "en", "en-GB", "en_US"),
    min_len: int = 1,
    max_len: int = 400,
) -> List[AlignedSentence]:
    """
    解析 TMX，直接输出 AlignedSentence 列表
    一个 <tu> = 一个对齐单元
    """
    if not os.path.exists(tmx_path):
        raise FileNotFoundError(f"TMX 文件不存在：{tmx_path}")

    tree = ET.parse(tmx_path)
    root = tree.getroot()

    cn_langs = {x.lower() for x in cn_langs}
    en_langs = {x.lower() for x in en_langs}

    aligned_units: List[AlignedSentence] = []

    for idx, tu in enumerate(root.iter("tu"), start=1):
        zh_text = ""
        en_text = ""
        project_name = ""

        # 1) 先提取 TU 级别的 project name
        for prop in tu.findall("prop"):
            prop_type = (prop.attrib.get("type", "") or "").strip()
            if prop_type == "x-Project Name:SingleString":
                project_name = normalize_sentence("".join(prop.itertext()))
                break


        for tuv in tu.findall("tuv"):
            lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "").lower()

            seg = tuv.find("seg")
            if seg is None:
                continue

            # 用 itertext 避免 TMX 内联标签造成文本丢失
            text = "".join(seg.itertext()).strip()
            text = normalize_sentence(text)

            if len(text) > max_len:
                text = text[:max_len]

            if lang in cn_langs:
                zh_text = text
            elif lang in en_langs:
                en_text = text

        if zh_text and len(zh_text) < min_len:
            zh_text = ""
        if en_text and len(en_text) < min_len:
            en_text = ""

        aligned_units.append(
            AlignedSentence(
                pair_key=f"tu_{idx}",
                zh_text=zh_text,
                en_text=en_text,
                project_name=project_name,
                metadata={
                    "source_type": "tmx",
                    "source_path": os.path.abspath(tmx_path),
                    "source_file": os.path.basename(tmx_path),
                    "tu_index": idx,
                    "cn_empty": (zh_text == ""),
                    "en_empty": (en_text == ""),
                    "missing": ("cn" if not zh_text else "") + ("en" if not en_text else "")
                }
            )
        )

    return aligned_units


def aligned_units_to_jsonl_rows(units: List[AlignedSentence]) -> List[Dict[str, Any]]:
    rows = []
    for u in units:
        rows.append({
            "pair_key": u.pair_key,
            "zh_text": u.zh_text,
            "en_text": u.en_text,
            "project_name": u.project_name,
            "metadata": u.metadata,
        })
    return rows


def jsonl_rows_to_aligned_units(rows: List[Dict[str, Any]]) -> List[AlignedSentence]:
    units = []
    for r in rows:
        units.append(
            AlignedSentence(
                pair_key=r["pair_key"],
                zh_text=r.get("zh_text", ""),
                en_text=r.get("en_text", ""),
                project_name=r.get("project_name", ""),
                metadata=r.get("metadata", {}),
            )
        )
    return units

def deduplicate_aligned_units(units: List[AlignedSentence]) -> List[AlignedSentence]: #tmx去重
    seen = set()
    deduped = []

    for u in units:
        key = (u.zh_text.strip(), u.en_text.strip())

        if key in seen:
            continue

        seen.add(key)
        deduped.append(u)

    return deduped

# =========================
# 6) 构建索引项 & BM25
# =========================
def build_index_items(aligned_units: List[AlignedSentence]) -> List[IndexItem]:
    items: List[IndexItem] = []
    for u in aligned_units:
        if u.zh_text.strip():
            items.append(
                IndexItem(
                    index_text=u.zh_text,
                    lang="zh",
                    aligned=u,
                    metadata=u.metadata
                )
            )
        if u.en_text.strip():
            items.append(
                IndexItem(
                    index_text=u.en_text,
                    lang="en",
                    aligned=u,
                    metadata=u.metadata
                )
            )
    return items


def build_bm25_index(items: List[IndexItem]) -> BM25Index:
    corpus_tokens = [tokenize(it.index_text) for it in items]
    bm25 = BM25Okapi(corpus_tokens)
    return BM25Index(items=items, corpus_tokens=corpus_tokens, bm25=bm25)


def detect_query_lang(query: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", query or "") else "en"


def retrieve(
    index: BM25Index,
    query: str,
    top_k: int = 5,
    lang_hint: Optional[str] = None
) -> List[Tuple[IndexItem, float]]:
    """
    lang_hint:
      - None：全量检索
      - "zh"/"en"：只从对应语言索引项里检索
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = index.bm25.get_scores(q_tokens)
    idxs = list(range(len(scores)))

    if lang_hint in ("zh", "en"):
        idxs = [i for i in idxs if index.items[i].lang == lang_hint]

    ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)[:top_k]
    return [(index.items[i], float(scores[i])) for i in ranked if scores[i] > 0]


# =========================
# 7) 索引持久化
# =========================
def save_index(index: BM25Index, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump(index, f)


def load_index(out_dir: str) -> Optional[BM25Index]:
    p = os.path.join(out_dir, "bm25_index.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# =========================
# 8) TMX -> cache + manifest + BM25
# =========================
def build_or_load_index_from_tmx_dir(
        tmx_dir: str,
        work_dir: str,
        force_rebuild: bool = False,
        min_len: int = 1,
        max_len: int = 400,
) -> BM25Index:
        ensure_dir(work_dir)

        manifest_path = os.path.join(work_dir, "manifest.json")
        cache_dir = os.path.join(work_dir, "cache")
        ensure_dir(cache_dir)

        #manifest = load_json(manifest_path, default={"files": {}})

        manifest = load_json(manifest_path, default={})
        files_meta = manifest.setdefault("files", {})

        tmx_files = discover_tmx_files(tmx_dir)

        if not tmx_files:
            raise RuntimeError(f"未发现 TMX 文件：{tmx_dir}")

        all_aligned_units: List[AlignedSentence] = []
        rebuilt_files = 0

        for tmx_path in tmx_files:

            digest = file_digest(tmx_path)
            rec = manifest["files"].get(tmx_path)

            cache_path = os.path.join(cache_dir, f"{digest}.jsonl")

            # ✅ 命中缓存
            if (not force_rebuild) and rec and rec.get("digest") == digest and os.path.exists(cache_path):
                rows = load_jsonl(cache_path)
                units = jsonl_rows_to_aligned_units(rows)

            else:
                print(f"🔄 解析 TMX：{os.path.basename(tmx_path)}")

                units = parse_tmx_to_aligned_units(
                    tmx_path=tmx_path,
                    min_len=min_len,
                    max_len=max_len,
                )

                save_jsonl(cache_path, aligned_units_to_jsonl_rows(units))

                manifest["files"][tmx_path] = {
                    "digest": digest,
                    "updated_at": int(time.time()),
                    "units": len(units),
                }

                rebuilt_files += 1

            all_aligned_units.extend(units)

        print(f"✅ 总对齐单元：{len(all_aligned_units)}")
        print(f"✅ 本次重建文件数：{rebuilt_files}")

        # ✅ 构建 BM25
        all_aligned_units = deduplicate_aligned_units(all_aligned_units)
        items = build_index_items(all_aligned_units)
        idx = build_bm25_index(items)

        save_index(idx, work_dir)

        manifest.update({
            "tmx_dir": os.path.abspath(tmx_dir),
            "file_count": len(tmx_files),
            "aligned_units": len(all_aligned_units),
            "index_items": len(items),
            "last_build_at": int(time.time())
        })

        save_json(manifest_path, manifest)

        return idx


# =========================
# 9) RAG：检索 -> 组上下文 -> 调 LLM
# =========================
class RAGAgent:
    def __init__(self, llm: HelloAgentsLLM, index: BM25Index):
        self.llm = llm
        self.index = index

    def build_messages(self, query: str, hits: List[Tuple[IndexItem, float]]) -> List[Dict[str, str]]:
        seen = set()
        ctx_blocks = []

        for it, score in hits:
            u = it.aligned
            m = u.metadata
            uid = (m.get("source_file"), m.get("tu_index"))
            if uid in seen:
                continue
            seen.add(uid)

            src_file = m.get("source_file", "")

            ctx_blocks.append(
                f"[{len(seen)}] score={score:.4f} "
                f"source={src_file} tu={m.get('tu_index')}\n"
                f"CN: {u.zh_text}\n"
                f"EN: {u.en_text}"
            )

            if len(seen) >= 8:
                break

        context = "\n\n".join(ctx_blocks) if ctx_blocks else "（未检索到相关内容）"

        sys_rules = (
            "你是一个基于本地 TMX 双语知识库回答问题的助手。\n"
            "回答要求：\n"
            "1) 优先、尽量只使用“检索上下文”作答，不要编造。\n"
            "2) 如果上下文不足以回答，直接说明“上下文不足”。\n"
            "3) 关键结论后标注引用编号，例如：[1][3]。\n"
            "4) 若问题为中文，请用中文回答；若问题为英文，请用英文回答。\n"
            "5) 如果问题涉及翻译、术语、双语对照，可优先利用中英配对信息回答。\n"
        )

        return [
            {"role": "system", "content": sys_rules},
            {"role": "system", "content": f"检索上下文：\n{context}"},
            {"role": "user", "content": query},
        ]

    def ask(self, query: str, top_k: int = 8, temperature: float = 0) -> Optional[str]:
        lang = detect_query_lang(query)
        hits = retrieve(self.index, query, top_k=top_k, lang_hint=lang)
        messages = self.build_messages(query, hits)
        return self.llm.think(messages=messages, temperature=temperature)


# =========================
# 10) Search 输出
# =========================
def print_hits(hits: List[Tuple[IndexItem, float]]):
    if not hits:
        print("（无匹配）")
        return

    seen = set()
    show = 0

    for it, score in hits:
        u = it.aligned
        m = u.metadata
        uid = (m.get("source_file"), m.get("tu_index"))
        if uid in seen:
            continue
        seen.add(uid)
        show += 1

        print(f"\n#{show} score={score:.4f} | source={m.get('source_file')} | tu={m.get('tu_index')}")
        print(f"CN: {u.zh_text}")
        print(f"EN: {u.en_text}")
        if u.project_name:
            print(f"PROJECT: {u.project_name}")


        if m.get("cn_empty") or m.get("en_empty"):
            print("⚠️ 该 TU 存在单边缺失")

        if show >= 10:
            break


# =========================
# 11) CLI
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TMX -> Sentence Alignment -> BM25 -> (Optional) RAG"
    )
    parser.add_argument("--tmx_dir", type=str, required=True, help="TMX 文件路径")
    parser.add_argument("--work_dir", type=str, default="./bm25_work", help="工作目录：缓存与索引存放位置")
    parser.add_argument("--top_k", type=int, default=8, help="检索返回条数")
    parser.add_argument("--force_rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--min_len", type=int, default=1, help="过滤过短句子")
    parser.add_argument("--max_len", type=int, default=400, help="过长句子截断长度")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "rag"],
        default="search",
        help="search=只检索；rag=检索+调用LLM回答"
    )
    args = parser.parse_args()

    index = build_or_load_index_from_tmx_dir(
        tmx_dir=args.tmx_dir,
        work_dir=args.work_dir,
        force_rebuild=args.force_rebuild,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    if args.mode == "rag":
        llm = HelloAgentsLLM()
        agent = RAGAgent(llm, index)
        print("\n🟦 进入 RAG 模式（输入 exit 退出）")
        while True:
            q = input("\nQuery> ").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                break
            agent.ask(q, top_k=args.top_k, temperature=0)
    else:
        print("\n🟩 进入 Search 模式（只检索，输入 exit 退出）")
        while True:
            q = input("\nQuery> ").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                break
            lang = detect_query_lang(q)
            hits = retrieve(index, q, top_k=args.top_k, lang_hint=lang)
            print_hits(hits)


if __name__ == "__main__":
    main()
