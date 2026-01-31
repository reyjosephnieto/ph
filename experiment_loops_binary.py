"""Binary loop rotation experiment for topological stenosis and splitting."""

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
GROUND_TRUTH_BETTI_1 = 5
GREEN_ANGLES = (0, 90, 180, 270)
RED_ANGLES = (45, 135, 225, 315)


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_base_image() -> np.ndarray:
    """Create the base image with a horizontal chain of loops.

    Returns
    -------
    numpy.ndarray
        Binary image containing five narrow rectangular loops.
    """
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    loop_width = 30
    loop_height = 5
    inner_width = 28
    inner_height = 1
    spacing = 20
    count = 5

    total_width = count * loop_width + (count - 1) * spacing
    start_x = int(round(CENTER[0] - total_width / 2))
    start_y = int(round(CENTER[1] - loop_height / 2))

    for i in range(count):
        x0 = start_x + i * (loop_width + spacing)
        y0 = start_y
        img[y0 : y0 + loop_height, x0 : x0 + loop_width] = 255

        inner_x0 = x0 + (loop_width - inner_width) // 2
        inner_y0 = y0 + (loop_height - inner_height) // 2
        img[
            inner_y0 : inner_y0 + inner_height,
            inner_x0 : inner_x0 + inner_width,
        ] = 0

    return img


def rotate_and_binarize(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image and apply a hard threshold.

    Parameters
    ----------
    img : numpy.ndarray
        Input binary image.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    numpy.ndarray
        Rotated, binarized image.
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


def betti_1_from_image(img: np.ndarray) -> int:
    """Compute Betti-1 from a binary image via cubical persistence.

    Parameters
    ----------
    img : numpy.ndarray
        Binary image.

    Returns
    -------
    int
        Number of 1-dimensional features with long persistence.
    """
    inverted = (255 - img).astype(np.float32)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=inverted)
    complex_.persistence()
    intervals = complex_.persistence_intervals_in_dimension(1)
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
        Angle and Betti-1 count.
    """
    cv2.setNumThreads(0)
    base = make_base_image()
    rotated = rotate_and_binarize(base, angle)
    betti_1 = betti_1_from_image(rotated)
    return angle, betti_1


def plot_results(angles: Iterable[float], betti_1: Iterable[int]) -> None:
    """Plot Betti-1 across rotation angles.

    Parameters
    ----------
    angles : iterable of float
        Rotation angles.
    betti_1 : iterable of int
        Betti-1 counts.
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.step(angles, betti_1, where="post")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Betti-1 Count")
    ax.set_title(r"Topological Shattering of $\mathbf{H}_1$ Features under Rotation")
    ax.axhline(
        y=GROUND_TRUTH_BETTI_1, color="gray", linestyle=":", linewidth=1
    )
    for x in RED_ANGLES:
        ax.axvline(x=x, color="red", linestyle="--", linewidth=1)
    for x in GREEN_ANGLES:
        ax.axvline(x=x, color="green", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loops_binary.png"), dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the binary loop shattering experiment."""
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
    output_path = os.path.join(OUTPUT_DIR, "loops_binary.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)

    angles_sorted = [row[0] for row in results]
    betti_sorted = [row[1] for row in results]
    plot_results(angles_sorted, betti_sorted)


if __name__ == "__main__":
    main()
