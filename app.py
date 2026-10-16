"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

A modern human-in-the-loop web interface that adapts to the confidence
bucket of each PR, integrates LangGraph checkpointing with AsyncSqliteSaver,
and logs every interaction to the SQLite audit trail.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    risk_level_for,
)
from exercises.exercise_4_audit import build_graph

# Ensure proper stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# ─── Custom CSS for Premium Design ─────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Styles & Font adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header badge */
    .hero-title {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        margin-right: 0.5rem;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Card styling */
    .card-container {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.4);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Comment item */
    .comment-card {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #3b82f6;
    }
    .comment-blocker { border-left-color: #ef4444; }
    .comment-issue { border-left-color: #f97316; }
    .comment-suggestion { border-left-color: #3b82f6; }
    .comment-nit { border-left-color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None


# ─── Database helper for recent sessions ───────────────────────────────────
async def fetch_recent_sessions():
    """Query recent threads from the SQLite audit_events table."""
    try:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 20
                """
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ─── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ HITL PR Review Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Intelligent Pull-Request Triaging with LangGraph Human-in-the-Loop & Audit Trail</div>',
    unsafe_allow_html=True,
)


# ─── Sidebar — recent sessions & replay ────────────────────────────────────
with st.sidebar:
    st.header("📋 Review Sessions")
    st.caption("Structured logs from SQLite (`audit_events`)")

    recent_sessions = asyncio.run(fetch_recent_sessions())

    if recent_sessions:
        for s in recent_sessions:
            t_id = s["thread_id"]
            short_id = t_id[:8]
            risk = s.get("worst_risk", "med")
            risk_class = f"badge-{risk}"
            pr_num = s["pr_url"].split("/")[-1] if "/" in s["pr_url"] else "?"

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f"**PR #{pr_num}** `<{short_id}>`  \n"
                    f"<span class='metric-badge {risk_class}'>{risk.upper()}</span> "
                    f"<span style='font-size:0.8rem; color:#94a3b8;'>{s['events']} events</span>",
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button("Inspect", key=f"sel_{t_id}"):
                    st.session_state.thread_id = t_id
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.rerun()
            st.divider()
    else:
        st.info("No recorded review sessions yet. Run your first review above!")

    st.markdown("---")
    st.caption(f"Database: `{db_path()}`")


# ─── Top Form — Select or Enter PR ─────────────────────────────────────────
st.markdown("### 🔍 Target Pull Request")

col_quick1, col_quick2, col_space = st.columns([1.5, 1.5, 3])
with col_quick1:
    if st.button("📌 Demo #1 (Task Priority)", use_container_width=True):
        st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
        st.rerun()
with col_quick2:
    if st.button("⚠️ Demo #2 (Auth & Sync)", use_container_width=True):
        st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
        st.rerun()

with st.form("start_form"):
    pr_url_input = st.text_input(
        "GitHub Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    col_submit, col_clear = st.columns([1, 4])
    with col_submit:
        submitted = st.form_submit_button("🚀 Start Review", type="primary", use_container_width=True)


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Trigger new run ───────────────────────────────────────────────────────
if submitted and pr_url_input:
    st.session_state.pr_url = pr_url_input.strip()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.last_action = None

    with st.spinner("Fetching PR metadata, unified diff, and invoking LLM analysis..."):
        try:
            result = asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id))
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.final = result
            st.rerun()
        except Exception as err:
            st.error(f"Error during agent invocation: {err}")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload.get("confidence", 0.65)
    risk = risk_level_for(conf)
    badge_class = f"badge-{risk}"

    st.markdown(
        f"""
        <div class="card-container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <h3 style="margin:0;">👤 Human Approval Requested</h3>
                <div>
                    <span class="metric-badge {badge_class}">RISK: {risk.upper()}</span>
                    <span class="metric-badge badge-info">CONFIDENCE: {conf:.0%}</span>
                </div>
            </div>
            <p style="color:#94a3b8; font-style:italic; margin-bottom:1rem;">{payload.get("confidence_reasoning", "")}</p>
            <div style="margin-bottom:1.5rem;">
                <h4 style="margin-bottom:0.5rem;">Summary of Changes</h4>
                <p>{payload.get("summary", "")}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💬 Proposed Review Comments")
    comments = payload.get("comments", [])
    if comments:
        for c in comments:
            sev = c.get("severity", "suggestion")
            st.markdown(
                f"""
                <div class="comment-card comment-{sev}">
                    <span class="metric-badge badge-{'high' if sev in ('blocker', 'issue') else 'med'}">{sev.upper()}</span>
                    <code>{c.get('file')}:{c.get('line') or '?'}</code>
                    <div style="margin-top:0.4rem;">{c.get('body')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No inline comments proposed.")

    with st.expander("📄 Unified Diff Preview", expanded=False):
        diff_text = payload.get("diff_preview", "")
        st.code(diff_text, language="diff")

    st.markdown("#### ✍️ Reviewer Decision")
    feedback = st.text_input(
        "Optional feedback or instructions for the author",
        placeholder="e.g. Looks good, approved with task priority schema update.",
        key="approval_feedback_input",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve & Post Review", type="primary", use_container_width=True):
            return {"choice": "approve", "feedback": feedback}
    with col2:
        if st.button("🚫 Reject Review", use_container_width=True):
            return {"choice": "reject", "feedback": feedback}
    with col3:
        if st.button("✏️ Request Edits", use_container_width=True):
            return {"choice": "edit", "feedback": feedback}

    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload.get("confidence", 0.50)
    risk = "high"

    st.markdown(
        f"""
        <div class="card-container" style="border-left: 6px solid #ef4444;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <h3 style="margin:0; color:#ef4444;">⚠️ Strong Escalation Required</h3>
                <div>
                    <span class="metric-badge badge-high">RISK: HIGH</span>
                    <span class="metric-badge badge-high">CONFIDENCE: {conf:.0%}</span>
                </div>
            </div>
            <p style="color:#f87171; font-style:italic;">{payload.get("confidence_reasoning", "")}</p>
            <p>{payload.get("summary", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_factors = payload.get("risk_factors", [])
    if risk_factors:
        st.markdown("#### 🚨 Identified Security & Architecture Risks")
        for rf in risk_factors:
            st.error(f"• {rf}")

    with st.expander("📄 Unified Diff Preview", expanded=False):
        st.code(payload.get("diff_preview", payload.get("summary", "")), language="diff")

    st.markdown("#### ❓ Reviewer Clarification Form")
    st.info("The agent needs answers to the following questions before synthesizing a final review:")

    questions = payload.get("questions", [])
    if not questions:
        questions = ["What is the primary intent of this PR?", "Are there migration or security concerns?"]

    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}: {q}**")
            answers[q] = st.text_area(
                label=f"Answer for Q{idx+1}",
                label_visibility="collapsed",
                placeholder="Enter technical explanation or mitigation...",
                key=f"esc_ans_{idx}",
            )
        submitted_answers = st.form_submit_button(
            "⚡ Submit Answers & Synthesize Refined Review",
            type="primary",
            use_container_width=True,
        )
        if submitted_answers:
            return answers

    return None


# ─── Render Active Interrupt (if any) ──────────────────────────────────────
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind", "approval_request")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming graph execution & synthesizing..."):
            try:
                res = asyncio.run(
                    run_graph(
                        st.session_state.pr_url,
                        st.session_state.thread_id,
                        resume_value=answer,
                    )
                )
                if "__interrupt__" in res:
                    st.session_state.interrupt_payload = res["__interrupt__"][0].value
                else:
                    st.session_state.interrupt_payload = None
                    st.session_state.final = res
                st.rerun()
            except Exception as err:
                st.error(f"Resume failed: {err}")


# ─── Render Final Result State (if completed) ──────────────────────────────
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "completed")
    analysis = final.get("analysis")
    conf = analysis.confidence if analysis else 1.0

    st.markdown("### 🏁 Review Execution Outcome")

    if action.startswith("auto") or action.startswith("committed"):
        st.success(
            f"**Action Completed:** `{action}` — Review comment successfully posted to GitHub PR!"
        )
    elif action == "rejected":
        st.warning(f"**Review Rejected:** `{action}` — No comment was posted.")
    else:
        st.info(f"**Result:** `{action}`")

    if analysis:
        st.markdown(
            f"""
            <div class="card-container">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4>Final Review Summary</h4>
                    <span class="metric-badge badge-low">FINAL CONFIDENCE: {conf:.0%}</span>
                </div>
                <p>{analysis.summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─── Interactive Audit Trail Replay ────────────────────────────────────
    if st.session_state.thread_id:
        with st.expander("📜 Session Audit Trail (Live Replay)", expanded=True):
            events = asyncio.run(replay_events(st.session_state.thread_id))
            if events:
                cols = st.columns([2, 2, 1, 1, 2, 1])
                cols[0].markdown("**Timestamp**")
                cols[1].markdown("**Action**")
                cols[2].markdown("**Confidence**")
                cols[3].markdown("**Risk**")
                cols[4].markdown("**Decision**")
                cols[5].markdown("**Time**")
                st.divider()

                for ev in events:
                    r_cols = st.columns([2, 2, 1, 1, 2, 1])
                    r_cols[0].text(ev["timestamp"][:19])
                    r_cols[1].markdown(f"`{ev['action']}`")
                    r_cols[2].text(f"{ev['confidence']:.0%}")
                    r_badge = f"badge-{ev['risk_level']}"
                    r_cols[3].markdown(
                        f"<span class='metric-badge {r_badge}'>{ev['risk_level'].upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    r_cols[4].markdown(f"**{ev['decision']}**")
                    r_cols[5].text(f"{ev['execution_time_ms']}ms")
                    if ev.get("reason"):
                        st.caption(f"↳ Reason: {ev['reason']}")
            else:
                st.caption("No events recorded for this session yet.")
