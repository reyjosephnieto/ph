"""Binary shattering experiment for rotation-induced topological instability."""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Tuple

import cv2
import gudhi
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = 1024
CENTER = (511.5, 511.5)
OUTPUT_DIR = os.path.join(".", "wobble")
GROUND_TRUTH_BETTI_0 = 30
GREEN_ANGLES = (0, 90, 180, 270)
RED_ANGLES = (45, 135, 225, 315)


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_base_image() -> np.ndarray:
    """Create the base binary image with a vertical string of pearls.

    Returns
    -------
    numpy.ndarray
        Base image with 30 distinct foreground pixels.
    """
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    start_y = 200
    x = 512
    for i in range(30):
        y = start_y + (i * 2)
        img[y, x] = 255
    return img


def rotate_and_binarize(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate and threshold the image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    numpy.ndarray
        Binarized rotated image.
    """
    matrix = cv2.getRotationMatrix2D(CENTER, angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(rotated > 127, 255, 0).astype(np.uint8)


def betti_0_from_image(img: np.ndarray) -> int:
    """Compute Betti-0 from a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        Binary image.

    Returns
    -------
    int
        Connected component count.
    """
    inverted = (255 - img).astype(np.float32)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=inverted)
    complex_.persistence()
    intervals = complex_.persistence_intervals_in_dimension(0)
    if intervals.size == 0:
        return 0
    return int(np.sum(np.isinf(intervals[:, 1]) | (intervals[:, 1] > 200)))


def process_angle(angle: float) -> Tuple[float, int]:
    """Process a single rotation angle.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    tuple of (float, int)
        Angle and Betti-0 count.
    """
    cv2.setNumThreads(0)
    base = make_base_image()
    rotated = rotate_and_binarize(base, angle)
    betti_0 = betti_0_from_image(rotated)
    return angle, betti_0


def plot_results(angles: Iterable[float], betti_0: Iterable[int]) -> None:
    """Plot Betti-0 against angle with reference lines.

    Parameters
    ----------
    angles : iterable of float
        Rotation angles.
    betti_0 : iterable of int
        Betti-0 counts.
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.step(angles, betti_0, where="post")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Betti-0 Count")
    ax.set_title(r"Topological Shattering of $\mathbf{H}_0$ Features under Rotation")
    ax.axhline(
        y=GROUND_TRUTH_BETTI_0, color="gray", linestyle=":", linewidth=1
    )
    for x in RED_ANGLES:
        ax.axvline(x=x, color="red", linestyle="--", linewidth=1)
    for x in GREEN_ANGLES:
        ax.axvline(x=x, color="green", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "shattering_binary.png"), dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the binary shattering experiment."""
    ensure_output_dir()

    angles = np.arange(0.0, 360.0 + 1e-9, 1.0)
    max_workers = max(1, (os.cpu_count() or 1) - 1)

    results: List[Tuple[float, int]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_angle, float(angle)): float(angle)
            for angle in angles
        }
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"completed {completed}/{total}")

    results.sort(key=lambda item: item[0])
    output_path = os.path.join(OUTPUT_DIR, "shattering_binary.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)

    angles_sorted = [row[0] for row in results]
    betti_sorted = [row[1] for row in results]
    plot_results(angles_sorted, betti_sorted)


if __name__ == "__main__":
    main()
