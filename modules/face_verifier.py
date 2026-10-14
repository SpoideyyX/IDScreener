"""
face_verifier.py

Phase 3 module — Facial biometric matching (ID photo vs. live selfie).

Approach
--------
1. Detect + crop the face region from the ID document image (the photo is
   usually a sub-region of the full document scan, so we detect-and-crop
   rather than assuming the whole image is a face).
2. Detect the face in the selfie image the same way.
3. Compute an embedding for each face using DeepFace (Facenet backend —
   lighter than ArcFace/VGG-Face, runs fine on CPU for a demo).
4. Compare embeddings with cosine distance and convert to a 0-1 similarity
   score using the configured threshold.

Why Facenet + cosine instead of ArcFace: ArcFace is more accurate on
large-scale benchmarks but is noticeably heavier to run on a CPU-only laptop
during a live demo. Facenet is a reasonable accuracy/speed trade-off for a
course project — mention this trade-off explicitly in your viva if asked
"why not ArcFace."
"""

import cv2
import numpy as np
from deepface import DeepFace

import config


class FaceNotDetectedError(Exception):
    """Raised when no face can be found in a given image."""


# --------------------------------------------------------------------------- #
# Face detection / cropping
# --------------------------------------------------------------------------- #

def detect_and_crop_face(image_path: str, padding_ratio: float = 0.25) -> np.ndarray:
    """Detect the largest face in an image and return a padded crop of it.

    Uses DeepFace's built-in detector backend (RetinaFace) rather than a
    separate OpenCV Haar cascade — this keeps detection and embedding on the
    same face-alignment convention, which noticeably improves match quality.
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
        )
    except ValueError as e:
        raise FaceNotDetectedError(f"No face detected in {image_path}: {e}")

    if not detections:
        raise FaceNotDetectedError(f"No face detected in {image_path}")

    # extract_faces returns normalized float images in [0,1]; convert back
    # to a standard uint8 BGR crop for consistent downstream display/storage.
    face_float = detections[0]["face"]
    face_uint8 = (face_float * 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
    return face_bgr


# --------------------------------------------------------------------------- #
# Embedding + comparison
# --------------------------------------------------------------------------- #

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def get_embedding(face_image: np.ndarray) -> np.ndarray:
    """Compute a DeepFace embedding for an already-cropped face image."""
    result = DeepFace.represent(
        img_path=face_image,
        model_name=config.FACE_MODEL_NAME,
        detector_backend="skip",   # already detected/cropped — skip re-detection
        enforce_detection=False,
    )
    return np.array(result[0]["embedding"])


def distance_to_similarity(distance: float, threshold: float = config.FACE_DISTANCE_THRESHOLD) -> float:
    """Map a raw cosine distance to a 0-1 'same person' similarity score,
    centered so that `distance == threshold` maps to 0.5. This gives a
    smooth score for the dashboard rather than a hard binary cutoff."""
    # Linear mapping around the threshold, clipped to [0, 1].
    similarity = 0.5 + (threshold - distance) / (2 * threshold)
    return float(np.clip(similarity, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Phase-1 interface contract
# --------------------------------------------------------------------------- #

def verify(id_image_path: str, selfie_path: str) -> dict:
    """Compare the face on an ID document against a live selfie.

    Returns
    -------
    dict with:
      score        : float in [0, 1] — similarity score (higher = more likely same person)
      distance     : float — raw cosine distance between embeddings
      id_face_crop : np.ndarray — the cropped face from the ID, for dashboard display
    """
    id_face_crop = detect_and_crop_face(id_image_path)
    selfie_face_crop = detect_and_crop_face(selfie_path)

    id_embedding = get_embedding(id_face_crop)
    selfie_embedding = get_embedding(selfie_face_crop)

    distance = _cosine_distance(id_embedding, selfie_embedding)
    score = distance_to_similarity(distance)

    return {
        "score": score,
        "distance": distance,
        "id_face_crop": id_face_crop,
        "selfie_face_crop": selfie_face_crop,  # extra, handy for the dashboard's side-by-side view
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare a face on an ID document against a selfie.")
    parser.add_argument("id_image", type=str)
    parser.add_argument("selfie_image", type=str)
    args = parser.parse_args()

    try:
        result = verify(args.id_image, args.selfie_image)
        print(f"Cosine distance : {result['distance']:.4f}")
        print(f"Similarity score: {result['score']:.3f}")
        print("Match: SAME PERSON (likely)" if result["score"] >= 0.5 else "Match: DIFFERENT PERSON (likely)")
    except FaceNotDetectedError as e:
        print(f"Error: {e}")
