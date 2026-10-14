"""
generate_synthetic_forgeries.py

Phase 2 deliverable — AI-Based Fake Identity and Document Screening System

Purpose
-------
Real leaked/forged ID datasets are (rightly) hard to get for a course project.
This script takes your "genuine" sample ID images and synthetically tampers
them in ways that mimic real-world forgery techniques, so you have a labeled
Genuine/Forged dataset to train and evaluate the ELA classifier locally.

Forgery types simulated (each is a real tampering technique):
  1. Text overlay tampering   — simulates editing a name/DOB/ID number field
  2. Photo splicing           — simulates swapping the ID photo with another
  3. Copy-move cloning        — simulates duplicating a region to cover/hide text
  4. Recompression artifact   — simulates a screenshot-and-re-save forgery
     (no visible edit, but a JPEG-generation-history inconsistency — a good
     "hard negative" so your classifier doesn't just learn to spot obvious edits)

Usage
-----
    python generate_synthetic_forgeries.py \
        --genuine_dir dataset/genuine \
        --output_dir dataset/forged \
        --per_image 4

Requirements: opencv-python, numpy, Pillow (already in requirements.txt)
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# Individual forgery operations
# --------------------------------------------------------------------------- #

def tamper_text_overlay(image: np.ndarray) -> np.ndarray:
    """Paste a new text string over part of the image, mimicking an edited
    name/DOB/ID field. Uses PIL for cleaner font rendering than cv2.putText."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    h, w = image.shape[:2]

    # Pick a plausible "text field" region (lower-left third, where names/DOB
    # usually sit on ID templates) and paint over it before writing new text —
    # this mimics how a forger blanks a field before typing a replacement.
    x = random.randint(int(w * 0.05), int(w * 0.35))
    y = random.randint(int(h * 0.55), int(h * 0.8))
    box_w, box_h = random.randint(120, 220), random.randint(20, 34)
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(245, 245, 240))

    fake_strings = ["JOHN A SMITH", "01/01/1999", "ID-88213X", "EXP 12/2030"]
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=box_h - 6)
    except IOError:
        font = ImageFont.load_default()
    draw.text((x + 4, y + 2), random.choice(fake_strings), fill=(20, 20, 20), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def tamper_photo_splice(image: np.ndarray) -> np.ndarray:
    """Simulate photo-swap fraud: cut a rectangular region (standing in for
    the ID photo slot), then paste back a resized/blurred/noised version of
    itself from elsewhere in the image — approximating the blending artifacts
    of a spliced-in photo without needing a second source image."""
    h, w = image.shape[:2]
    result = image.copy()

    # Assume photo sits in the left third of the document — common ID layout.
    px, py = int(w * 0.05), int(h * 0.15)
    pw, ph = int(w * 0.28), int(h * 0.55)
    pw, ph = max(pw, 10), max(ph, 10)

    # Source a donor patch from elsewhere in the image, resize to fit,
    # then blend it in with a soft-edged mask (splices rarely have hard edges
    # after a forger blurs the boundary).
    donor_x = random.randint(0, max(1, w - pw))
    donor_y = random.randint(0, max(1, h - ph))
    donor = image[donor_y:donor_y + ph, donor_x:donor_x + pw]
    donor = cv2.resize(donor, (pw, ph))
    donor = cv2.GaussianBlur(donor, (3, 3), 0)

    mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.ellipse(mask, (pw // 2, ph // 2), (pw // 2 - 3, ph // 2 - 3), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    roi = result[py:py + ph, px:px + pw].astype(np.float32)
    blended = (roi * (1 - mask_3ch) + donor.astype(np.float32) * mask_3ch).astype(np.uint8)
    result[py:py + ph, px:px + pw] = blended

    return result


def tamper_copy_move(image: np.ndarray) -> np.ndarray:
    """Copy a small patch and paste it elsewhere on the same image — the
    classic 'copy-move' forgery used to hide a security feature or duplicate
    a stamp/signature."""
    h, w = image.shape[:2]
    result = image.copy()
    patch_w, patch_h = random.randint(40, 90), random.randint(20, 45)

    src_x = random.randint(0, max(1, w - patch_w))
    src_y = random.randint(0, max(1, h - patch_h))
    patch = image[src_y:src_y + patch_h, src_x:src_x + patch_w]

    dst_x = random.randint(0, max(1, w - patch_w))
    dst_y = random.randint(0, max(1, h - patch_h))
    result[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch

    return result


def tamper_recompression(image: np.ndarray) -> np.ndarray:
    """No visible edit — just repeatedly re-encode at low JPEG quality to
    mimic a 'screenshot of a screenshot' forgery. This is intentionally a
    HARD example: your classifier should ideally still flag inconsistent
    compression history, but don't be surprised if this class has lower
    detection accuracy — that's a good, honest talking point for your report."""
    result = image
    for quality in (60, 40, 55):
        _, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


FORGERY_FUNCTIONS = {
    "text_overlay": tamper_text_overlay,
    "photo_splice": tamper_photo_splice,
    "copy_move": tamper_copy_move,
    "recompression": tamper_recompression,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def generate_forgeries(genuine_dir: str, output_dir: str, per_image: int, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    genuine_paths = sorted(
        [p for p in Path(genuine_dir).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not genuine_paths:
        raise FileNotFoundError(
            f"No images found in '{genuine_dir}'. Add sample ID images there first "
            f"(see the dataset sourcing notes in README / Phase 2 notes)."
        )

    os.makedirs(output_dir, exist_ok=True)
    forgery_names = list(FORGERY_FUNCTIONS.keys())
    manifest_rows = ["filename,source_image,forgery_type,label"]
    count = 0

    for src_path in genuine_paths:
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"  [skip] couldn't read {src_path}")
            continue

        # Cycle through forgery types so the dataset stays balanced across
        # techniques rather than randomly clumping.
        chosen_types = random.sample(
            forgery_names * ((per_image // len(forgery_names)) + 1), per_image
        )

        for i, forgery_type in enumerate(chosen_types):
            forged = FORGERY_FUNCTIONS[forgery_type](image.copy())

            # Apply one more forgery type ~30% of the time, to simulate
            # documents with more than one tampered region — more realistic
            # and pushes the classifier past trivially separable cases.
            if random.random() < 0.3:
                second_type = random.choice([t for t in forgery_names if t != forgery_type])
                forged = FORGERY_FUNCTIONS[second_type](forged)
                forgery_type = f"{forgery_type}+{second_type}"

            out_name = f"{src_path.stem}_forged_{i}.jpg"
            out_path = Path(output_dir) / out_name
            cv2.imwrite(str(out_path), forged, [cv2.IMWRITE_JPEG_QUALITY, 92])

            manifest_rows.append(f"{out_name},{src_path.name},{forgery_type},FORGED")
            count += 1

        print(f"  [{src_path.name}] -> {per_image} forged variants")

    manifest_path = Path(output_dir) / "forged_manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows))

    print(f"\nDone. Generated {count} forged images into '{output_dir}'.")
    print(f"Manifest written to '{manifest_path}'.")
    print(
        "\nReminder: also write a matching manifest row (label=GENUINE) for each "
        "original file in genuine_dir when you build the training CSV in Phase 3 —"
        " ela_detector.py's training script expects a unified genuine+forged manifest."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic forged ID images for training.")
    parser.add_argument("--genuine_dir", default="dataset/genuine", help="Folder of clean/genuine ID images")
    parser.add_argument("--output_dir", default="dataset/forged", help="Where forged images are written")
    parser.add_argument("--per_image", type=int, default=4, help="Number of forged variants per genuine image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating synthetic forgeries from '{args.genuine_dir}' ...")
    generate_forgeries(args.genuine_dir, args.output_dir, args.per_image, args.seed)


if __name__ == "__main__":
    main()
