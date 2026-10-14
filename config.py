"""Central configuration for the fake ID screening pipeline.
Every module in `modules/` imports from here so thresholds/weights live in
exactly one place — this is also the file to point to in your viva when
asked "how did you choose these numbers?"
"""

# --- Image processing ---
IMG_SIZE = (224, 224)          # standard resize for classifier input
ELA_QUALITY = 90               # JPEG re-save quality for ELA (lower = more sensitive)
ELA_SCALE = 15                 # multiplier to make the ELA diff visible

# --- Trust scoring weights (must sum to 1.0) ---
WEIGHT_ELA = 0.45              # tampering detection weighted highest — strongest
                                # single fraud signal available on a course-project dataset
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
FACE_DISTANCE_THRESHOLD = 0.40      # cosine distance below this => same person
                                     # (Facenet + cosine typical operating point)

# --- OCR field validation ---
MIN_OCR_CONFIDENCE = 0.35           # EasyOCR results below this confidence are discarded
CURRENT_YEAR = 2026                 # used for expiry / DOB sanity checks

# --- Paths ---
DATASET_DIR = "dataset"
GENUINE_DIR = f"{DATASET_DIR}/genuine"
FORGED_DIR = f"{DATASET_DIR}/forged"
ELA_MAPS_DIR = f"{DATASET_DIR}/ela_maps"
MODEL_PATH = "models/ela_classifier.joblib"
