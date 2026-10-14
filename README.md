# AI-Based Fake Identity and Document Screening System
### Phase 1 — Architecture & Environment Setup

## 1. Directory Layout

```
fake_id_screening/
├── app.py                        # Streamlit entrypoint (Phase 4)
├── requirements.txt
├── README.md
├── config.py                     # thresholds, weights, paths — single source of truth
│
├── modules/
│   ├── __init__.py
│   ├── ela_detector.py           # ELA image transform + classifier train/infer (Phase 3)
│   ├── ocr_validator.py          # EasyOCR extraction + regex/rule validation (Phase 3)
│   ├── face_verifier.py          # Face crop from ID + DeepFace similarity (Phase 3)
│   └── trust_scorer.py           # Weighted fusion of the three signals (Phase 3)
│
├── dataset/
│   ├── genuine/                  # your source "clean" ID images go here
│   ├── forged/                   # synthetic tampered images land here (Phase 2 script writes here)
│   ├── ela_maps/                 # cached ELA outputs, used as classifier training features
│   └── generate_synthetic_forgeries.py   # Phase 2 deliverable
│
├── models/
│   └── ela_classifier.joblib     # trained RandomForest (or CNN weights) saved here
│
└── notebooks/
    └── eda_preprocessing.ipynb   # class balance, resolution stats, ELA visualization
```

**Why this layout works for a 2nd-year ML course project:**
- Each pipeline stage is an independent, testable module with one clear responsibility — you can demo/unit-test `ela_detector.py` alone without OCR or face code running.
- `config.py` centralizes magic numbers (score weights, thresholds, image sizes) so your viva answer to "how did you pick these weights?" points to one file, not scattered constants.
- `dataset/` cleanly separates real vs. synthetic-forged data so your evaluation metrics are honest about what's real vs. augmented.

## 2. Environment Setup

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Two setup gotchas worth knowing before you hit them:**
1. **DeepFace's first run downloads model weights** (~100MB) from GitHub releases — do this once with internet access before your offline demo/viva, or the dashboard will hang on first face-match.
2. **EasyOCR's first run also downloads detection/recognition weights** (~50-100MB). Same advice — run a warm-up script the night before your demo.

Warm-up script (`warmup.py`, run once):
```python
import easyocr
from deepface import DeepFace

print("Downloading EasyOCR weights...")
easyocr.Reader(['en'])

print("Downloading DeepFace weights...")
DeepFace.build_model("Facenet")

print("Warm-up complete.")
```

## 3. Pipeline Architecture (end-to-end data flow)

```
                    ┌──────────────────┐
   ID Document ────►│  1. ELA Detector  │───► authenticity_score (0-1)
   (image)          │  (tamper regions) │
                    └──────────────────┘

                    ┌──────────────────┐
   ID Document ────►│  2. OCR Validator │───► field_confidence_score (0-1)
   (image)          │  (fields + regex) │      + extracted fields dict
                    └──────────────────┘

   ID Document  ┐   ┌──────────────────┐
   Selfie       ├──►│ 3. Face Verifier  │───► face_similarity_score (0-1)
   (image)      ┘   │ (crop + compare)  │
                    └──────────────────┘

              all 3 scores
                    │
                    ▼
          ┌────────────────────┐
          │  4. Trust Scorer    │───► Authenticity Score (0-100%)
          │  (weighted fusion)  │───► Verdict: GENUINE / SUSPICIOUS / FRAUDULENT
          └────────────────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ 5. Streamlit UI     │
          │ (upload, visualize, │
          │  display verdict)   │
          └────────────────────┘
```

## 4. Config file (create this now — every later module imports it)

```python
# config.py
"""Central configuration for the fake ID screening pipeline."""

# --- Image processing ---
IMG_SIZE = (224, 224)          # standard resize for classifier input
ELA_QUALITY = 90               # JPEG re-save quality for ELA (lower = more sensitive)
ELA_SCALE = 15                 # multiplier to make the ELA diff visible

# --- Trust scoring weights (must sum to 1.0) ---
WEIGHT_ELA = 0.45              # tampering detection is weighted highest — it's the
                                # strongest fraud signal for a course-project dataset
WEIGHT_OCR = 0.25              # field consistency — catches logic errors forgers miss
WEIGHT_FACE = 0.30             # biometric match — catches identity-swap fraud

assert abs(WEIGHT_ELA + WEIGHT_OCR + WEIGHT_FACE - 1.0) < 1e-6

# --- Verdict thresholds (on the final 0-100 score) ---
THRESHOLD_GENUINE = 75         # >= this => GENUINE
THRESHOLD_SUSPICIOUS = 45      # >= this and < GENUINE => SUSPICIOUS / MANUAL REVIEW
                                # below THRESHOLD_SUSPICIOUS => FRAUDULENT

# --- Face verification ---
FACE_DISTANCE_METRIC = "cosine"     # "cosine" or "euclidean_l2"
FACE_MODEL_NAME = "Facenet"         # lighter than ArcFace/VGG-Face, faster on CPU

# --- Paths ---
DATASET_DIR = "dataset"
GENUINE_DIR = f"{DATASET_DIR}/genuine"
FORGED_DIR = f"{DATASET_DIR}/forged"
ELA_MAPS_DIR = f"{DATASET_DIR}/ela_maps"
MODEL_PATH = "models/ela_classifier.joblib"
```

**On the weights (75/25/30 split → normalized to 0.45/0.25/0.30):** this is a defensible
starting point, not a tuned result — be upfront about that in your viva. Frame it as
"initial weights based on which signal is hardest to fake" and mention that a proper
ablation study (Phase 5 covers how to discuss this) would justify them empirically.

## 5. What each module will expose (interface contract, so Phase 3 slots in cleanly)

| Module | Main function | Returns |
|---|---|---|
| `ela_detector.py` | `analyze(image_path) -> dict` | `{"score": float, "ela_image": np.ndarray, "verdict": str}` |
| `ocr_validator.py` | `extract_and_validate(image_path) -> dict` | `{"fields": dict, "score": float, "errors": list}` |
| `face_verifier.py` | `verify(id_image_path, selfie_path) -> dict` | `{"score": float, "distance": float, "id_face_crop": np.ndarray}` |
| `trust_scorer.py` | `compute_trust_score(ela_result, ocr_result, face_result) -> dict` | `{"final_score": float, "verdict": str, "breakdown": dict}` |

This contract is what lets you build and unit-test each module independently, then wire
them together in `app.py` in Phase 4 with minimal glue code.

---
Next: **Phase 2 — Synthetic dataset & forgery generation script** (below).
