"""
app.py

Phase 4 — Streamlit dashboard for the AI-Based Fake Identity and
Document Screening System with Live Camera Biometric Verification.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from modules import ela_detector, ocr_validator, face_verifier, trust_scorer

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Fake ID Screening System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI-Based Fake Identity & Document Screening System")
st.caption(
    "Upload an ID document and capture a live selfie to run tampering detection, "
    "OCR field validation, and facial biometric matching."
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def save_uploaded_file(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile / Camera buffer to a temp path so our
    modules can read it with cv2/PIL."""
    name = getattr(uploaded_file, "name", "live_capture.jpg")
    suffix = os.path.splitext(name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_gauge(score: float, verdict: str) -> go.Figure:
    """Trust-score gauge: green/amber/red bands matching verdict thresholds."""
    color_map = {
        "GENUINE": "#2ecc71",
        "SUSPICIOUS / MANUAL REVIEW REQUIRED": "#f39c12",
        "FRAUDULENT": "#e74c3c",
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        title={"text": "Authenticity Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_map.get(verdict, "#3498db")},
            "steps": [
                {"range": [0, config.THRESHOLD_SUSPICIOUS], "color": "#fdecea"},
                {"range": [config.THRESHOLD_SUSPICIOUS, config.THRESHOLD_GENUINE], "color": "#fef5e7"},
                {"range": [config.THRESHOLD_GENUINE, 100], "color": "#eafaf1"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


VERDICT_STYLE = {
    "GENUINE": ("✅", "success"),
    "SUSPICIOUS / MANUAL REVIEW REQUIRED": ("⚠️", "warning"),
    "FRAUDULENT": ("🚫", "error"),
}


def render_verdict_banner(verdict: str) -> None:
    icon, kind = VERDICT_STYLE.get(verdict, ("ℹ️", "info"))
    message = f"{icon}  **Verdict: {verdict}**"
    getattr(st, kind)(message)


# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("1. Document Input")
    id_file = st.file_uploader("ID Document (required)", type=["jpg", "jpeg", "png"])

    st.header("2. Biometric Selfie")
    selfie_source = st.radio(
        "Choose selfie input method:",
        options=["Live Camera Capture", "Upload Image File", "Skip Biometric Verification"],
        index=0,
    )

    selfie_file = None
    if selfie_source == "Live Camera Capture":
        st.caption("Look directly at the camera and snap a photo.")
        selfie_file = st.camera_input("Take Live Selfie")
    elif selfie_source == "Upload Image File":
        selfie_file = st.file_uploader("Upload Selfie Photo", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.caption(
        f"Weights: ELA {config.WEIGHT_ELA:.0%} · OCR {config.WEIGHT_OCR:.0%} · "
        f"Face {config.WEIGHT_FACE:.0%}"
    )
    st.caption(
        f"Thresholds: ≥{config.THRESHOLD_GENUINE} GENUINE · "
        f"≥{config.THRESHOLD_SUSPICIOUS} SUSPICIOUS · below is FRAUDULENT"
    )

    run_button = st.button("🔍 Run Screening", type="primary", disabled=id_file is None)

# --- Initial Guard Checks ---
if id_file is None:
    st.info("👈 Upload an ID document in the sidebar to begin.")
    st.stop()

if selfie_source == "Live Camera Capture" and selfie_file is None:
    st.info("👈 Please snap a picture using the sidebar camera, then click **Run Screening**.")
    st.stop()

if not run_button:
    st.info("👈 Press **Run Screening** in the sidebar to analyze.")
    st.stop()


# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #

id_path = save_uploaded_file(id_file)
selfie_path = save_uploaded_file(selfie_file) if selfie_file is not None else None

col_doc, col_selfie = st.columns(2)
with col_doc:
    st.subheader("ID Document")
    st.image(id_path, width="stretch")
with col_selfie:
    st.subheader("Live Selfie / Portrait")
    if selfie_path:
        st.image(selfie_path, width="stretch")
    else:
        st.info("No selfie provided — face-match step will be skipped.")

st.markdown("---")

# --- 1. ELA tampering detection --- #
st.header("1️⃣ Tampering Detection (ELA)")
ela_result = None
try:
    if not os.path.exists(config.MODEL_PATH):
        st.warning(
            f"No trained model found at `{config.MODEL_PATH}`. Run "
            f"`python -m modules.ela_detector --train` first. Showing the ELA "
            f"heatmap only, without a tampering score."
        )
        ela_image = ela_detector.compute_ela(id_path)
        col1, col2 = st.columns(2)
        with col1:
            st.image(id_path, caption="Original", width="stretch")
        with col2:
            st.image(ela_image, caption="ELA Heatmap (untrained)", width="stretch")
    else:
        ela_result = ela_detector.analyze(id_path)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(id_path, caption="Original", width="stretch")
        with col2:
            st.image(ela_result["ela_image"], caption="ELA Heatmap", width="stretch")
        with col3:
            st.metric("Genuine Probability", f"{ela_result['score']:.1%}")
            st.metric("Verdict", ela_result["verdict"])
            if ela_result["verdict"] == "LIKELY_TAMPERED":
                st.caption(
                    "🔴 Bright patches in the heatmap indicate regions with an "
                    "inconsistent compression history — likely edited."
                )
except Exception as e:
    st.error(f"ELA analysis failed: {e}")

# --- 2. OCR extraction + validation --- #
st.header("2️⃣ OCR Field Extraction & Validation")
ocr_result = None
try:
    with st.spinner("Extracting text and validating fields..."):
        ocr_result = ocr_validator.extract_and_validate(id_path)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Extracted Fields**")
        fields = ocr_result["fields"]
        st.write(f"- Name: `{fields['name'] or 'Not found'}`")
        st.write(f"- Date of Birth: `{fields['dob'] or 'Not found'}`")
        st.write(f"- ID Number: `{fields['id_number'] or 'Not found'}`")
        st.write(f"- Expiry: `{fields['expiry'] or 'Not found'}`")
    with col2:
        st.metric("OCR / Validation Score", f"{ocr_result['score']:.1%}")
        if ocr_result["errors"]:
            st.write("**Validation Issues:**")
            for err in ocr_result["errors"]:
                st.write(f"- ⚠️ {err}")
        else:
            st.success("All field validation checks passed.")
except Exception as e:
    st.error(f"OCR validation failed: {e}")

# --- 3. Face verification --- #
face_result = None
if selfie_path:
    st.header("3️⃣ Facial Biometric Matching")
    try:
        with st.spinner("Detecting faces and computing biometric distance..."):
            face_result = face_verifier.verify(id_path, selfie_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(bgr_to_rgb(face_result["id_face_crop"]), caption="ID Photo (cropped)", width="stretch")
        with col2:
            st.image(bgr_to_rgb(face_result["selfie_face_crop"]), caption="Live Capture (cropped)", width="stretch")
        with col3:
            st.metric("Similarity Score", f"{face_result['score']:.1%}")
            st.metric("Cosine Distance", f"{face_result['distance']:.3f}")
            match_label = "Likely SAME person" if face_result["score"] >= 0.5 else "Likely DIFFERENT person"
            st.write(f"**{match_label}**")
    except face_verifier.FaceNotDetectedError as e:
        st.error(f"Biometric matching failed: {e}")
    except Exception as e:
        st.error(f"Biometric matching failed unexpectedly: {e}")
else:
    st.header("3️⃣ Facial Biometric Matching")
    st.info("Skipped — no live capture or selfie was provided.")

# --- 4. Trust score fusion --- #
st.markdown("---")
st.header("🎯 Final Authenticity Verdict")

if ela_result is not None and ocr_result is not None:
    final = trust_scorer.compute_trust_score(ela_result, ocr_result, face_result)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(render_gauge(final["final_score"], final["verdict"]), width="stretch")
    with col2:
        render_verdict_banner(final["verdict"])
        st.write("**Score Breakdown:**")
        for component, data in final["breakdown"].items():
            if component == "override_applied":
                st.warning(data)
                continue
            if data["raw_score"] is None:
                st.write(f"- {component.upper()}: not evaluated (weight redistributed)")
            else:
                st.write(
                    f"- {component.upper()}: raw score {data['raw_score']:.1%} "
                    f"× weight {data['weight']:.0%} = **{data['contribution']:.1f} pts**"
                )
else:
    st.warning(
        "Final score requires both a trained ELA model and successful OCR extraction — "
        "resolve any errors above and re-run."
    )

# --------------------------------------------------------------------------- #
# Cleanup temp files
# --------------------------------------------------------------------------- #

for p in (id_path, selfie_path):
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass
