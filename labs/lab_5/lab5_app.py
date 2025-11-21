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


# ---------- Face detection + drawing ----------
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

    # Open with Pillow for further processing
    try:
        with Image.open(in_path) as img:
            img_converted = img.convert("RGB")  # for compatibility with OpenCV
    except Exception as e:
        print("Failed to open image with Pillow:", e)
        sys.exit(3)

    # Apply cascade to detect faces and save the image with rectangles
    try:
        result_img, face_count = detect_faces_and_draw(img_converted)
        result_img.save(out_image_path, format="JPEG")
        print(f"Image with outlined faces saved to: {out_image_path}")
        print(f"Faces found: {face_count}")
    except Exception as e:
        print("Error detecting faces or saving image:", e)
        sys.exit(4)


if __name__ == "__main__":
    main()
