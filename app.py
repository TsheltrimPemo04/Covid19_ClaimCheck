import os
import json
import time
import subprocess
import streamlit as st

# ---------------------------------------------------------------
# Auto-build retrieval index if missing
# ---------------------------------------------------------------

if not os.path.exists("sentence_embeddings.npy"):

    with st.spinner("Building retrieval index... This may take several minutes on first launch."):

        subprocess.run(["python", "build_index.py"])

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Claim Check",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Cached model loader — runs ONCE per Streamlit process
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and evidence index (one-time, ~30s)...")
def load_verifier():
    import verify_claim as vc      # heavy imports happen here
    return vc

@st.cache_data(show_spinner=False)
def load_index_meta():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "index_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
st.markdown("""
<style>
.verdict-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    color: white;
    margin: 0.5rem 0 1.5rem 0;
}
.verdict-true   { background: linear-gradient(135deg, #16a34a, #15803d); }
.verdict-false  { background: linear-gradient(135deg, #dc2626, #991b1b); }
.verdict-nei    { background: linear-gradient(135deg, #6b7280, #4b5563); }
.verdict-card h2 { color: white; margin: 0; font-size: 1.8rem; }
.verdict-card p  { color: white; margin: 0.3rem 0 0 0; opacity: 0.95; }

.evidence-card {
    border-left: 4px solid #ddd;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    background: #f9fafb;
    border-radius: 4px;
}
.evidence-supports { border-left-color: #16a34a; }
.evidence-refutes  { border-left-color: #dc2626; }
.evidence-neutral  { border-left-color: #9ca3af; }
.evidence-meta { font-size: 0.82rem; color: #555; margin-top: 0.4rem; }
.score-pill {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    font-size: 0.75rem; margin-right: 4px; background:#e5e7eb; color:#374151;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Sidebar — index info + parameters
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Checks COVID-19 claims against a corpus of peer-reviewed research "
        "abstracts using semantic retrieval + a FEVER-trained NLI model."
    )
    st.markdown("---")
    st.markdown("### Index")
    meta = load_index_meta()
    if meta:
        st.markdown(f"**Documents:** {meta['n_documents']:,}")
        st.markdown(f"**Sentences:** {meta['n_sentences']:,}")
        st.markdown(f"**Embedding model:**  \n`{meta['embedding_model']}`")
    else:
        st.error("No index found. Run `python build_index.py` first.")

    st.markdown("---")
    st.markdown("### Retrieval settings")
    top_k = st.slider("Evidence sentences retrieved", 5, 30, 15, 1)
    st.markdown("Set higher for noisier claims, lower for speed.")


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🦠 COVID-19 Claim Check")
st.markdown(
    "Enter a claim about COVID-19 and the system will retrieve evidence from "
    "research abstracts, classify each piece as **supporting / refuting / neutral**, "
    "and produce an overall verdict with citations."
)

# ---------------------------------------------------------------
# Example claims
# ---------------------------------------------------------------
EXAMPLES = [
    "Wearing face masks reduces the transmission of SARS-CoV-2.",
    "Hydroxychloroquine is an effective treatment for COVID-19.",
    "Vitamin D deficiency is associated with worse COVID-19 outcomes.",
    "5G networks cause COVID-19.",
    "SARS-CoV-2 was created in a laboratory.",
]

if "claim_text" not in st.session_state:
    st.session_state.claim_text = ""

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
        st.session_state.claim_text = ex

claim = st.text_area(
    "Your claim",
    value=st.session_state.claim_text,
    height=80,
    placeholder="e.g. Vaccines cause autism in COVID-19 patients",
)

go = st.button("🔍 Check claim", type="primary", use_container_width=True)


# ---------------------------------------------------------------
# Verification
# ---------------------------------------------------------------
def verdict_card(verdict: str, n_sup: int, n_ref: int):
    cls = {"TRUE": "verdict-true", "FALSE": "verdict-false",
            "NOT ENOUGH INFO": "verdict-nei"}[verdict]
    icon = {"TRUE": "✅", "FALSE": "❌", "NOT ENOUGH INFO": "❓"}[verdict]
    label = {"TRUE": "Likely TRUE",
            "FALSE": "Likely FALSE",
            "NOT ENOUGH INFO": "Not enough evidence"}[verdict]
    st.markdown(f"""
        <div class="verdict-card {cls}">
            <h2>{icon} {label}</h2>
            <p>{n_sup} supporting · {n_ref} refuting evidence sentence(s) found in the corpus.</p>
        </div>
    """, unsafe_allow_html=True)


def _safe_str(v):
    """Coerce DataFrame values (which may be NaN floats) to clean strings."""
    if v is None:
        return ""
    if isinstance(v, float):
        # math.isnan would also work; this handles NaN without an extra import
        if v != v:
            return ""
    return str(v).strip()

def render_evidence(items, group_class):
    for e in items:
        title = _safe_str(e.get("title")) or "(untitled)"
        doi = _safe_str(e.get("doi"))
        date = _safe_str(e.get("publish_time"))[:10]
        journal = _safe_str(e.get("journal"))
        link = f"https://doi.org/{doi}" if doi else None

        meta_bits = []
        if journal: meta_bits.append(journal)
        if date and date != "NaT": meta_bits.append(date)
        if doi:     meta_bits.append(f"[DOI]({link})" if link else doi)
        meta_line = " · ".join(meta_bits)

        st.markdown(f"""
            <div class="evidence-card {group_class}">
                <em>"{e['sentence']}"</em>
                <div class="evidence-meta">
                    <span class="score-pill">sim {e['similarity']:.2f}</span>
                    <span class="score-pill">support {e['p_support']:.2f}</span>
                    <span class="score-pill">refute {e['p_refute']:.2f}</span>
                    <br><strong>{title[:200]}</strong><br>{meta_line}
                </div>
            </div>
        """, unsafe_allow_html=True)


if go and claim.strip():
    if not meta:
        st.error("Cannot run — index not built. Run `python build_index.py` first.")
    else:
        vc = load_verifier()
        with st.spinner(f"Retrieving evidence and running NLI on top-{top_k} sentences..."):
            t0 = time.time()
            result = vc.verify(claim.strip(), top_k=top_k)
            dt = time.time() - t0

        verdict_card(result["verdict"], result["n_supporting"], result["n_refuting"])
        st.caption(f"Decision in {dt:.1f}s · claim: _{result['claim']}_")

        ev = result["evidence"]
        sup = [e for e in ev if e["stance"] == "SUPPORTS"]
        ref = [e for e in ev if e["stance"] == "REFUTES"]
        neu = [e for e in ev if e["stance"] == "NEUTRAL"]

        tab1, tab2, tab3 = st.tabs([
            f"Supporting ({len(sup)})",
            f"Refuting ({len(ref)})",
            f"Neutral / off-topic ({len(neu)})",
        ])
        with tab1:
            if sup: render_evidence(sup, "evidence-supports")
            else:   st.info("No supporting evidence found.")
        with tab2:
            if ref: render_evidence(ref, "evidence-refutes")
            else:   st.info("No refuting evidence found.")
        with tab3:
            render_evidence(neu, "evidence-neutral")

elif go:
    st.warning("Please enter a claim.")

st.markdown("---")
st.caption(
    "Built on the CORD-19 research abstracts corpus. "
    "Retrieval: sentence-transformers · Stance classification: DeBERTa-v3 fine-tuned on MNLI+FEVER+ANLI. "
    "Results reflect what's in the corpus, not absolute scientific truth."
)
