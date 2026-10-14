"""
ela_detector.py

Phase 3 module — Error Level Analysis (ELA) based tampering detector.

What ELA actually measures
---------------------------
A genuine JPEG has a roughly *uniform* compression error across the whole
image, because every region went through the same single compression pass.
When someone edits a region (pastes text, splices a photo) and re-saves the
file, that edited region gets a *fresh* compression pass laid on top of the
original one — so it has a different, usually higher, error level than the
untouched parts of the image. Re-saving the image at a known JPEG quality
and diffing it against the original makes that inconsistency visible as
bright patches in the edited regions.

This module provides:
  1. `compute_ela(image_path)`      -> raw ELA difference image
  2. `extract_features(ela_image)`  -> fixed-length feature vector for ML
  3. `train_classifier(...)`        -> trains + saves a RandomForest on a
                                        genuine/forged manifest
  4. `analyze(image_path)`          -> the Phase-1 interface contract:
                                        {"score": float, "ela_image": np.ndarray, "verdict": str}
"""

import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import config


# --------------------------------------------------------------------------- #
# 1. ELA computation
# --------------------------------------------------------------------------- #

def compute_ela(image_path: str, quality: int = config.ELA_QUALITY, scale: int = config.ELA_SCALE) -> np.ndarray:
    """Re-save the image at a fixed JPEG quality and return the amplified
    absolute difference between the original and the re-saved version."""
    original = Image.open(image_path).convert("RGB")

    tmp_path = "_ela_tmp.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    resaved = Image.open(tmp_path)

    orig_arr = np.asarray(original, dtype=np.int16)
    resaved_arr = np.asarray(resaved, dtype=np.int16)

    diff = np.abs(orig_arr - resaved_arr)
    diff = np.clip(diff * scale, 0, 255).astype(np.uint8)

    os.remove(tmp_path)
    return diff


# --------------------------------------------------------------------------- #
# 2. Feature extraction from the ELA image
# --------------------------------------------------------------------------- #

def extract_features(ela_image: np.ndarray) -> np.ndarray:
    """Turn the ELA diff image into a compact feature vector.

    Rather than feeding raw pixels into a RandomForest (which would need a
    huge amount of data to generalize), we compute summary statistics that
    directly encode "how uniform is the error across the image" — which is
    exactly the property that separates genuine from tampered documents.
    """
    gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY) if ela_image.ndim == 3 else ela_image
    gray = cv2.resize(gray, config.IMG_SIZE)

    # Global statistics — a tampered region raises the max/std even if the
    # mean barely moves, since most of the image is still "clean".
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    max_val = float(np.max(gray))
    p95_val = float(np.percentile(gray, 95))

    # Block-wise variance: split into a grid and look at how much the error
    # level varies block-to-block. Genuine images -> low variance across
    # blocks. Tampered images -> a few blocks (the edited region) spike.
    grid = 8
    h, w = gray.shape
    block_means = []
    for i in range(grid):
        for j in range(grid):
            block = gray[
                i * h // grid:(i + 1) * h // grid,
                j * w // grid:(j + 1) * w // grid,
            ]
            block_means.append(np.mean(block))
    block_means = np.array(block_means)
    block_variance = float(np.var(block_means))
    block_max = float(np.max(block_means))

    # Edge density from the ELA map — spliced boundaries tend to leave sharp
    # edges even after blurring, unlike a document's naturally smooth print.
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    return np.array([
        mean_val, std_val, max_val, p95_val,
        block_variance, block_max, edge_density,
    ])


FEATURE_NAMES = [
    "mean", "std", "max", "p95",
    "block_variance", "block_max", "edge_density",
]


# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #

def build_dataset(genuine_dir: str, forged_dir: str):
    """Walk genuine_dir and forged_dir, compute ELA + features for every
    image, and return (X, y) ready for sklearn."""
    X, y = [], []

    genuine_paths = list(Path(genuine_dir).glob("*"))
    forged_paths = list(Path(forged_dir).glob("*.jpg")) + list(Path(forged_dir).glob("*.png"))

    for p in genuine_paths:
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        ela = compute_ela(str(p))
        X.append(extract_features(ela))
        y.append(0)  # 0 = genuine

    for p in forged_paths:
        ela = compute_ela(str(p))
        X.append(extract_features(ela))
        y.append(1)  # 1 = forged

    if not X:
        raise ValueError(
            "No training images found. Populate dataset/genuine and run "
            "dataset/generate_synthetic_forgeries.py to populate dataset/forged first."
        )

    return np.array(X), np.array(y)


def train_classifier(genuine_dir: str = config.GENUINE_DIR,
                      forged_dir: str = config.FORGED_DIR,
                      model_path: str = config.MODEL_PATH,
                      test_size: float = 0.2,
                      random_state: int = 42) -> RandomForestClassifier:
    """Train a RandomForest on ELA-derived features and save it to disk."""
    print("Building feature dataset from genuine + forged images...")
    X, y = build_dataset(genuine_dir, forged_dir)
    print(f"Dataset: {len(y)} samples ({np.sum(y == 0)} genuine, {np.sum(y == 1)} forged)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",   # handles any genuine/forged imbalance
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["genuine", "forged"]))

    print("\nFeature importances:")
    for name, importance in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:16s} {importance:.3f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

    return clf


# --------------------------------------------------------------------------- #
# 4. Inference — the Phase-1 interface contract
# --------------------------------------------------------------------------- #

_model_cache = None


def _load_model(model_path: str = config.MODEL_PATH) -> RandomForestClassifier:
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run train_classifier() first "
                f"(see `python ela_detector.py --train`)."
            )
        _model_cache = joblib.load(model_path)
    return _model_cache


def analyze(image_path: str, model_path: str = config.MODEL_PATH) -> dict:
    """Run ELA + classifier on a single image.

    Returns
    -------
    dict with:
      score   : float in [0, 1] — probability the image is GENUINE (not tampered)
      ela_image : np.ndarray — the visual ELA map, for display in the dashboard
      verdict : "LIKELY_GENUINE" or "LIKELY_TAMPERED"
    """
    ela_image = compute_ela(image_path)
    features = extract_features(ela_image).reshape(1, -1)

    clf = _load_model(model_path)
    # predict_proba[:, 0] = probability of class 0 (genuine)
    genuine_probability = float(clf.predict_proba(features)[0][0])

    verdict = "LIKELY_GENUINE" if genuine_probability >= 0.5 else "LIKELY_TAMPERED"

    return {
        "score": genuine_probability,
        "ela_image": ela_image,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run the ELA tampering classifier.")
    parser.add_argument("--train", action="store_true", help="Train the classifier from dataset/")
    parser.add_argument("--infer", type=str, default=None, help="Path to an image to analyze")
    args = parser.parse_args()

    if args.train:
        train_classifier()
    elif args.infer:
        result = analyze(args.infer)
        print(f"Score (genuine probability): {result['score']:.3f}")
        print(f"Verdict: {result['verdict']}")
    else:
        parser.print_help()
