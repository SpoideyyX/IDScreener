"""
trust_scorer.py

Phase 3 module — Multi-modal fraud risk & trust scoring engine.

Combines the three independent signals:
  - ELA authenticity score   (modules/ela_detector.py)
  - OCR field-validity score (modules/ocr_validator.py)
  - Face similarity score    (modules/face_verifier.py)

into one weighted Authenticity Score (0-100%) and a verdict.

Design note for your report/viva: this is a simple weighted linear fusion,
not a learned meta-classifier. That's a deliberate, defensible scope choice
for a 2nd-year project — a learned fusion model would need a labeled dataset
of full end-to-end cases (ELA+OCR+face outcomes with ground-truth fraud
labels), which doesn't exist for a course project. Weighted fusion is
transparent, easy to justify field-by-field in a viva, and easy to re-tune
by editing config.py without retraining anything.
"""

import config


def compute_trust_score(ela_result: dict, ocr_result: dict, face_result: dict | None = None) -> dict:
    """Fuse the three module outputs into a final authenticity score.

    Parameters
    ----------
    ela_result  : output of ela_detector.analyze()          — needs "score"
    ocr_result  : output of ocr_validator.extract_and_validate() — needs "score"
    face_result : output of face_verifier.verify(), or None if no selfie was
                  provided (e.g. document-only screening mode). When None,
                  the ELA and OCR weights are renormalized to fill the gap
                  rather than silently scoring the missing signal as 0.

    Returns
    -------
    dict with:
      final_score : float in [0, 100]
      verdict     : "GENUINE" | "SUSPICIOUS / MANUAL REVIEW REQUIRED" | "FRAUDULENT"
      breakdown   : dict of each component's contribution, for the dashboard's
                    score-breakdown chart
    """
    ela_score = ela_result["score"]
    ocr_score = ocr_result["score"]

    if face_result is not None:
        face_score = face_result["score"]
        w_ela, w_ocr, w_face = config.WEIGHT_ELA, config.WEIGHT_OCR, config.WEIGHT_FACE
    else:
        # No selfie provided — redistribute the face weight proportionally
        # across ELA and OCR so the score isn't unfairly dragged down by a
        # signal that was never available.
        face_score = None
        total = config.WEIGHT_ELA + config.WEIGHT_OCR
        w_ela = config.WEIGHT_ELA / total
        w_ocr = config.WEIGHT_OCR / total
        w_face = 0.0

    weighted_ela = w_ela * ela_score
    weighted_ocr = w_ocr * ocr_score
    weighted_face = w_face * face_score if face_score is not None else 0.0

    final_score = round((weighted_ela + weighted_ocr + weighted_face) * 100, 1)

    if final_score >= config.THRESHOLD_GENUINE:
        verdict = "GENUINE"
    elif final_score >= config.THRESHOLD_SUSPICIOUS:
        verdict = "SUSPICIOUS / MANUAL REVIEW REQUIRED"
    else:
        verdict = "FRAUDULENT"

    breakdown = {
        "ela": {"raw_score": ela_score, "weight": w_ela, "contribution": round(weighted_ela * 100, 1)},
        "ocr": {"raw_score": ocr_score, "weight": w_ocr, "contribution": round(weighted_ocr * 100, 1)},
        "face": {
            "raw_score": face_score,
            "weight": w_face,
            "contribution": round(weighted_face * 100, 1),
        },
    }

    # Hard override: even a high fused score shouldn't read as GENUINE if the
    # tampering detector is highly confident of forgery — a single strong
    # red flag should not be averaged away by two clean signals. This kind
    # of override rule is a good thing to mention in your viva as a
    # deliberate design choice, not an oversight in the weighting.
    if ela_score < 0.15 and verdict == "GENUINE":
        verdict = "SUSPICIOUS / MANUAL REVIEW REQUIRED"
        breakdown["override_applied"] = "ELA score critically low — verdict downgraded from GENUINE"

    return {
        "final_score": final_score,
        "verdict": verdict,
        "breakdown": breakdown,
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint — quick manual sanity check without running the full pipeline
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Example: a document with strong ELA/OCR signals but no selfie provided
    example_ela = {"score": 0.92}
    example_ocr = {"score": 0.81}

    result = compute_trust_score(example_ela, example_ocr, face_result=None)
    print("Document-only screening (no selfie):")
    print(f"  Final score: {result['final_score']}")
    print(f"  Verdict    : {result['verdict']}")
    print(f"  Breakdown  : {result['breakdown']}")

    print()

    example_face = {"score": 0.35}  # weak face match
    result2 = compute_trust_score(example_ela, example_ocr, face_result=example_face)
    print("Full pipeline (document + selfie), weak face match:")
    print(f"  Final score: {result2['final_score']}")
    print(f"  Verdict    : {result2['verdict']}")
    print(f"  Breakdown  : {result2['breakdown']}")
