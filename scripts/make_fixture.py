"""Generate deterministic synthetic smoke inputs; no depth ground truth."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('runs/fixture'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 255, 640, dtype=np.uint8)
    y = np.linspace(0, 255, 480, dtype=np.uint8)
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :, 0] = x[None]
    image[:, :, 1] = y[:, None]
    image[:, :, 2] = 128
    cv2.rectangle(image, (80, 120), (300, 350), (40, 200, 230), -1)
    cv2.circle(image, (460, 230), 85, (220, 80, 40), -1)
    if not cv2.imwrite(str(args.output_dir/'synthetic.png'), image):
        raise RuntimeError('Could not write fixture')
    writer = cv2.VideoWriter(str(args.output_dir/'synthetic.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15., (640, 480))
    if not writer.isOpened():
        raise RuntimeError('MJPG codec unavailable')
    try:
        for shift in range(30):
            writer.write(np.roll(image, shift*2, axis=1))
    finally:
        writer.release()
    print(args.output_dir)


if __name__ == '__main__':
    main()
