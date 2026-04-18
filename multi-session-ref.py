"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, 스트리밍 답변.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# --- 경로 및 환경 변수 ---
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)


def _resolve_log_dir() -> Path | None:
    """로컬은 프로젝트 logs/, Streamlit Cloud 등은 /tmp 하위로 쓴다 (읽기 전용 FS 대비)."""
    candidates = [
        ROOT / "logs",
        Path(tempfile.gettempdir()) / "multisession_rag_logs",
    ]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    return None


def _build_log_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_dir = _resolve_log_dir()
    if not log_dir:
        return handlers
    log_file = log_dir / f"multisession_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
    except OSError:
        pass
    return handlers


for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_build_log_handlers(),
)
log = logging.getLogger("multi_session_rag")

MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_EMBED = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 8

# Streamlit Cloud Secrets와 로컬 .env 모두에서 읽는 키
_ENV_KEYS = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")


def apply_streamlit_secrets_to_environ() -> None:
    """Streamlit Cloud의 Secrets를 os.environ에 반영한다. load_dotenv 이후 호출되면 값을 덮어쓴다."""
    try:
        sec = st.secrets
    except Exception:
        return
    for key in _ENV_KEYS:
        try:
            val = sec.get(key)
        except Exception:
            val = None
        if val is not None and str(val).strip():
            os.environ[key] = str(val).strip()


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]+~~", "", text)
    text = re.sub(r"^[\s]*[-_=]{3,}[\s]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def env_ok() -> tuple[bool, str]:
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        return (
            False,
            "다음 환경 변수가 없습니다: "
            + ", ".join(missing)
            + " (로컬: 프로젝트 루트 .env / Streamlit Cloud: 앱 설정 → Secrets)",
        )
    return True, ""


@st.cache_resource
def get_supabase() -> Client | None:
    ok, _ = env_ok()
    if not ok:
        return None
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


def get_embeddings() -> OpenAIEmbeddings | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)


def get_llm(streaming: bool = True, temperature: float = 0.7) -> ChatOpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, streaming=streaming)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "work_session_id" not in st.session_state:
        st.session_state.work_session_id = None
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None
    if "retrieval_session_id" not in st.session_state:
        st.session_state.retrieval_session_id = None
    if "show_vectordb" not in st.session_state:
        st.session_state.show_vectordb = False
    if "pending_upload_hashes" not in st.session_state:
        st.session_state.pending_upload_hashes = set()


def ensure_work_session(sb: Client) -> str | None:
    if st.session_state.work_session_id:
        return st.session_state.work_session_id
    row = (
        sb.table("chat_sessions")
        .insert({"title": "(임시) 새 대화"})
        .execute()
    )
    if not row.data:
        return None
    sid = row.data[0]["id"]
    st.session_state.work_session_id = sid
    st.session_state.retrieval_session_id = sid
    return sid


def list_sessions(sb: Client) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,created_at,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def load_messages_for_session(sb: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        sb.table("chat_messages")
        .select("role,content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    rows = res.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def sync_messages_to_db(sb: Client, session_id: str, messages: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    for m in messages:
        sb.table("chat_messages").insert(
            {"session_id": session_id, "role": m["role"], "content": m["content"]}
        ).execute()


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_with_rpc(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    q_emb = embeddings.embed_query(query)
    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": q_emb,
                "match_session_id": session_id,
                "match_count": k,
            },
        ).execute()
        rows = res.data or []
    except Exception as e:
        log.warning("RPC match_vector_documents 실패, 로컬 필터로 대체: %s", e)
        rows = _retrieve_fallback(sb, q_emb, session_id, k)

    docs = []
    for r in rows:
        docs.append(
            Document(
                page_content=r.get("content", ""),
                metadata={"file_name": r.get("file_name", ""), "id": r.get("id")},
            )
        )
    return docs


def _retrieve_fallback(
    sb: Client, query_emb: list[float], session_id: str, k: int
) -> list[dict[str, Any]]:
    res = (
        sb.table("vector_documents")
        .select("id,content,file_name,embedding")
        .eq("session_id", session_id)
        .execute()
    )
    rows = res.data or []
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        emb = r.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        if not emb or not isinstance(emb, list):
            continue
        scored.append((cosine_sim(query_emb, [float(x) for x in emb]), r))
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, r in scored[:k]:
        out.append(
            {
                "id": r["id"],
                "content": r["content"],
                "file_name": r["file_name"],
                "similarity": 1.0,
            }
        )
    return out


def insert_vectors_direct(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    file_name: str,
    texts: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    extra_meta = extra_meta or {}
    for i in range(0, len(texts), BATCH_EMBED):
        batch = texts[i : i + BATCH_EMBED]
        vecs = embeddings.embed_documents(batch)
        rows = []
        for j, (txt, vec) in enumerate(zip(batch, vecs)):
            meta = {
                **extra_meta,
                "chunk_index": i + j,
                "file_name": file_name,
            }
            rows.append(
                {
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": txt,
                    "embedding": vec,
                    "metadata": meta,
                }
            )
        sb.table("vector_documents").insert(rows).execute()


def process_pdfs(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    files: list[Any],
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    processed: list[str] = []
    for uf in files:
        raw = uf.getvalue()
        h = hashlib.sha256(raw).hexdigest()
        key = f"{session_id}:{h}"
        if key in st.session_state.pending_upload_hashes:
            processed.append(f"{uf.name} (이미 임베딩됨, 스킵)")
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        splits = splitter.split_documents(docs)
        texts = [d.page_content for d in splits if d.page_content.strip()]
        if not texts:
            processed.append(f"{uf.name} (텍스트 없음)")
            continue
        insert_vectors_direct(sb, embeddings, session_id, uf.name, texts, {"sha256": h})
        st.session_state.pending_upload_hashes.add(key)
        processed.append(uf.name)
    return processed


def generate_session_title(first_q: str, first_a: str) -> str:
    llm = get_llm(streaming=False, temperature=0.3)
    if not llm:
        return "새 세션"
    prompt = (
        "다음 첫 질문과 첫 답변을 바탕으로 대화 제목을 한국어로 짧게(40자 이내) 지으세요. "
        "따옴표나 '제목:' 같은 접두어 없이 제목만 출력하세요.\n\n"
        f"질문: {first_q}\n\n답변: {first_a[:1200]}"
    )
    out = llm.invoke(prompt)
    title = (out.content or "").strip().split("\n")[0][:80]
    return title or "새 세션"


def generate_followup_questions(answer_text: str) -> str:
    llm = get_llm(streaming=False, temperature=0.5)
    if not llm:
        return ""
    prompt = (
        "다음 답변을 읽고 사용자가 이어서 물어보면 좋을 질문을 정확히 3개만 한국어로 제시하세요. "
        "각 줄에 번호(1. 2. 3.)만 붙이고 한 줄에 한 질문씩.\n\n"
        f"답변:\n{answer_text[:4000]}"
    )
    out = llm.invoke(prompt)
    return (out.content or "").strip()


def build_rag_prompt(docs: list[Document], user_q: str) -> str:
    ctx = "\n\n".join(
        f"[출처: {d.metadata.get('file_name', '')}]\n{d.page_content}" for d in docs
    )
    return (
        "참고 문서를 바탕으로 답하세요. 문서에 없는 내용은 추측하지 말고 모른다고 하세요.\n\n"
        f"참고:\n{ctx}\n\n사용자 질문:\n{user_q}"
    )


def stream_chat(
    llm: ChatOpenAI,
    system_text: str,
    history: list[dict[str, str]],
    user_q: str,
) -> Generator[str, None, None]:
    msgs = [SystemMessage(content=system_text)]
    for m in history[:-1]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_q))
    for chunk in llm.stream(msgs):
        if chunk.content:
            yield chunk.content


def inject_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div.stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    logo_path = ROOT / "logo.png"
    col1, col2, _ = st.columns([1, 3, 1])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with col2:
        st.markdown(
            """
            <div style="text-align:center;">
            <span style="font-size:4rem !important; font-weight:700;">
            <span style="color:#1f77b4 !important;">멀티세션</span>
            <span style="color:#ffd700 !important;"> RAG 챗봇</span>
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="멀티세션 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    apply_streamlit_secrets_to_environ()
    inject_css()
    init_state()

    ok, env_msg = env_ok()
    sb = get_supabase()
    embeddings = get_embeddings()
    llm = get_llm(streaming=True)

    render_header()

    if not ok or sb is None:
        st.error(env_msg)
        st.stop()
    if embeddings is None or llm is None:
        st.error("OPENAI_API_KEY가 없어 임베딩/채팅을 사용할 수 없습니다.")
        st.stop()

    sessions = list_sessions(sb)
    sid_labels = {s["id"]: f"{s['title'][:60]} — {str(s['id'])[:8]}…" for s in sessions}
    ids = [s["id"] for s in sessions]
    st.session_state["_sb_client"] = sb

    def _on_session_pick() -> None:
        client: Client = st.session_state["_sb_client"]
        sel = st.session_state.get("session_pick")
        if not sel:
            return
        st.session_state.messages = load_messages_for_session(client, sel)
        st.session_state.retrieval_session_id = sel
        st.session_state.work_session_id = sel
        st.session_state.selected_session_id = sel

    with st.sidebar:
        st.markdown("### LLM 모델")
        st.radio("모델", [MODEL_NAME], index=0, disabled=True, label_visibility="collapsed")
        st.caption(f"고정 모델: {MODEL_NAME}")

        st.markdown("### 세션 관리")
        if ids:
            default_idx = 0
            pref = st.session_state.work_session_id or st.session_state.selected_session_id
            if pref in ids:
                default_idx = ids.index(pref)
            st.selectbox(
                "세션 선택",
                options=ids,
                format_func=lambda x: sid_labels.get(x, x),
                index=default_idx,
                key="session_pick",
                on_change=_on_session_pick,
            )
            st.session_state.selected_session_id = st.session_state.session_pick
        else:
            st.caption("저장된 세션이 없습니다.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션저장"):
                _action_save_session(sb)
        with c2:
            if st.button("세션로드"):
                _action_load_session(sb)
        c3, c4 = st.columns(2)
        with c3:
            if st.button("세션삭제"):
                _action_delete_session(sb)
        with c4:
            if st.button("화면초기화"):
                _action_reset_screen(sb)

        if st.button("vectordb"):
            st.session_state.show_vectordb = True

        st.markdown("### PDF 업로드")
        uploads = st.file_uploader(
            "PDF (다중)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기") and uploads:
            ws = ensure_work_session(sb)
            if ws:
                names = process_pdfs(sb, embeddings, ws, list(uploads))
                sync_messages_to_db(sb, ws, st.session_state.messages)
                st.success("처리됨: " + ", ".join(names))
                st.rerun()

        st.text(
            f"작업 세션 ID: {st.session_state.work_session_id or '—'}\n"
            f"검색(RAG) 세션 ID: {st.session_state.retrieval_session_id or '—'}\n"
            f"대화 메시지 수: {len(st.session_state.messages)}"
        )

    if st.session_state.show_vectordb:
        rid = st.session_state.retrieval_session_id or st.session_state.work_session_id
        st.subheader("Vector DB 파일 목록")
        if rid:
            res = (
                sb.table("vector_documents")
                .select("file_name")
                .eq("session_id", rid)
                .execute()
            )
            names = sorted({r["file_name"] for r in (res.data or [])})
            if names:
                for n in names:
                    st.write(f"- {n}")
            else:
                st.info("이 세션에 저장된 벡터가 없습니다.")
        else:
            st.info("세션이 없습니다.")
        if st.button("닫기"):
            st.session_state.show_vectordb = False
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]))

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ws = ensure_work_session(sb)
        rid = st.session_state.retrieval_session_id or ws
        docs: list[Document] = []
        if rid:
            docs = retrieve_with_rpc(sb, embeddings, rid, prompt)

        system_base = (
            "당신은 친절한 도우미입니다. 답변은 한국어 존댓말로 하고, "
            "# ## ### 마크다운 제목으로 구조화하세요. 구분선(---)과 취소선은 쓰지 마세요."
        )
        if docs:
            ctx_block = build_rag_prompt(docs, prompt)
            system_text = system_base + "\n\n" + ctx_block
        else:
            system_text = system_base

        assistant_placeholder = st.empty()
        full = []
        with st.chat_message("assistant"):
            for piece in stream_chat(llm, system_text, st.session_state.messages, prompt):
                full.append(piece)
                assistant_placeholder.markdown(remove_separators("".join(full)))
            body = "".join(full)
            fu = generate_followup_questions(body)
            if fu:
                body = body + "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n" + fu
            assistant_placeholder.markdown(remove_separators(body))
            st.session_state.messages.append({"role": "assistant", "content": body})

        if ws:
            sync_messages_to_db(sb, ws, st.session_state.messages)
        st.rerun()


def _action_save_session(sb: Client) -> None:
    msgs = st.session_state.messages
    if len(msgs) < 2:
        st.sidebar.warning("저장하려면 최소 한 번의 질문과 답변이 필요합니다.")
        return
    first_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
    first_a = ""
    for i, m in enumerate(msgs):
        if m["role"] == "user" and m["content"] == first_q:
            for j in range(i + 1, len(msgs)):
                if msgs[j]["role"] == "assistant":
                    first_a = msgs[j]["content"]
                    break
            break
    if not first_q or not first_a:
        st.sidebar.warning("첫 질문/답변을 찾을 수 없습니다.")
        return
    title = generate_session_title(first_q, first_a)
    work_id = st.session_state.work_session_id
    if not work_id:
        st.sidebar.error("작업 세션이 없습니다.")
        return
    ins = sb.table("chat_sessions").insert({"title": title}).execute()
    if not ins.data:
        st.sidebar.error("세션 INSERT 실패")
        return
    new_id = ins.data[0]["id"]
    for m in msgs:
        sb.table("chat_messages").insert(
            {"session_id": new_id, "role": m["role"], "content": m["content"]}
        ).execute()
    sb.table("vector_documents").update({"session_id": new_id}).eq(
        "session_id", work_id
    ).execute()
    sb.table("chat_messages").delete().eq("session_id", work_id).execute()
    sb.table("chat_sessions").delete().eq("id", work_id).execute()
    row = sb.table("chat_sessions").insert({"title": "(임시) 새 대화"}).execute()
    if row.data:
        st.session_state.work_session_id = row.data[0]["id"]
        st.session_state.retrieval_session_id = st.session_state.work_session_id
    st.session_state.messages = []
    st.session_state.pending_upload_hashes = set()
    st.sidebar.success(f"새 세션으로 저장됨: {title}")
    st.rerun()


def _action_load_session(sb: Client) -> None:
    sel = st.session_state.get("session_pick") or st.session_state.selected_session_id
    if not sel:
        st.sidebar.warning("불러올 세션을 선택하세요.")
        return
    st.session_state.messages = load_messages_for_session(sb, sel)
    st.session_state.retrieval_session_id = sel
    st.session_state.work_session_id = sel
    st.sidebar.success("세션을 불러왔습니다.")
    st.rerun()


def _action_delete_session(sb: Client) -> None:
    sel = st.session_state.get("session_pick") or st.session_state.selected_session_id
    if not sel:
        st.sidebar.warning("삭제할 세션을 선택하세요.")
        return
    sb.table("chat_sessions").delete().eq("id", sel).execute()
    if st.session_state.work_session_id == sel:
        st.session_state.work_session_id = None
    if st.session_state.retrieval_session_id == sel:
        st.session_state.retrieval_session_id = None
    st.session_state.messages = []
    st.session_state.selected_session_id = None
    if "session_pick" in st.session_state:
        del st.session_state.session_pick
    st.sidebar.success("세션이 삭제되었습니다.")
    st.rerun()


def _action_reset_screen(sb: Client) -> None:
    st.session_state.messages = []
    st.session_state.pending_upload_hashes = set()
    row = sb.table("chat_sessions").insert({"title": "(임시) 새 대화"}).execute()
    if row.data:
        st.session_state.work_session_id = row.data[0]["id"]
        st.session_state.retrieval_session_id = st.session_state.work_session_id
    st.sidebar.success("화면을 초기화했습니다.")
    st.rerun()


if __name__ == "__main__":
    main()
