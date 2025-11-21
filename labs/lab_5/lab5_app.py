#!/usr/bin/env python3
"""
lab5_app.py

Usage:
    python lab5_app.py new_york.jpeg

Result:
 - new_york_faces.jpg  (with faces outlined)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image, ExifTags, ImageOps
import cv2
import numpy as np


# ---------- JPEG validation & EXIF extraction ----------
def is_jpeg_magic_bytes(path: str) -> bool:
    """Check JPEG SOI marker (FF D8) at the file start."""
    try:
        with open(path, "rb") as f:
            start = f.read(2)
            return start == b"\xff\xd8"
    except Exception:
        return False


def pil_verify_jpeg(path: str) -> bool:
    """Use Pillow.verify() to check file integrity."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def exif_to_dict(pil_img: Image.Image) -> Dict[str, Any]:
    """Convert PIL EXIF to a JSON-serializable dict with human-readable tags."""
    out: Dict[str, Any] = {}
    try:
        exif = pil_img.getexif()
    except Exception:
        exif = None

    if not exif:
        return {}

    for tag_id, value in exif.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                value = repr(value)
        if tag == "GPSInfo" and isinstance(value, dict):
            gps = {}
            for t, v in value.items():
                subtag = ExifTags.GPSTAGS.get(t, t)
                gps[subtag] = v
            out[tag] = gps
        else:
            out[tag] = value
    return out


def apply_exif_orientation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF orientation (uses Pillow helper)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img


# ---------- Face detection + drawing (UNCHANGED) ----------
def detect_faces_and_draw(pil_img: Image.Image) -> Tuple[Image.Image, int]:
    """
    Detect frontal face projections using a Haar Cascade and draw red rectangles (BGR: (0,0,255)).
    Returns the updated PIL Image and the number of faces found.
    """
    # Convert PIL -> OpenCV (BGR)
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Cascade path in the OpenCV package
    # access to cv2.data.haarcascades is valid at runtime but some static checkers
    # may not recognize 'data' on the cv2 module; ignore the attribute-defined warning
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]
    if not os.path.exists(cascade_path):
        raise RuntimeError(f"Haar cascade not found at {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)
    # detectMultiScale parameters — adjustable if needed
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=10, minSize=(15, 15))

    # Draw rectangles
    for (x, y, w, h) in faces:
        # red rectangle, thickness 2 (BGR)
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert back to PIL
    result_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return result_pil, len(faces)


def main():
    if len(sys.argv) < 2:
        print("Usage: python lab5_app.py <image.jpg>")
        sys.exit(1)

    in_path = sys.argv[1]
    p = Path(in_path)
    if not p.is_file():
        print("File not found:", in_path)
        sys.exit(2)

    # Result filenames
    base = p.stem
    out_image_path = p.with_name(f"{base}_faces{p.suffix}")

    # 1) Validate JPEG
    print("Checking JPEG magic bytes...")
    if not is_jpeg_magic_bytes(in_path):
        print("Not a JPEG file (missing SOI marker)")
        sys.exit(3)

    print("Pillow verify()... ")
    if not pil_verify_jpeg(in_path):
        print("Pillow verify failed — file may be corrupted")
        sys.exit(4)

    # 2) Open with Pillow for EXIF and orientation
    try:
        with Image.open(in_path) as img:
            exif = exif_to_dict(img)
            img_oriented = apply_exif_orientation(img.convert("RGB"))
    except Exception as e:
        print("Failed to open image with Pillow:", e)
        sys.exit(5)

    # Print EXIF metadata to console (user requested no JSON file saving)
    if exif:
        print("EXIF metadata:")
        for k, v in exif.items():
            print(f"  {k}: {v}")
    else:
        print("No EXIF metadata found.")

    # 3) Detect faces and draw
    try:
        result_img, face_count = detect_faces_and_draw(img_oriented)
        result_img.save(out_image_path, format="JPEG")
        print(f"Image with outlined faces saved to: {out_image_path}")
        print(f"Faces found: {face_count}")
    except Exception as e:
        print("Error detecting faces or saving image:", e)
        sys.exit(6)


if __name__ == "__main__":
    main()
