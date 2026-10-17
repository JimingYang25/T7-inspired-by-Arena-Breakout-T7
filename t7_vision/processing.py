"""V6 input/output contract, independent of PyTorch."""
import cv2
import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess_bgr(frame, size_hw=(240, 320), mean=MEAN, std=STD):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
        raise ValueError('Expected a nonempty uint8 HWC BGR frame')
    h, w = (int(v) for v in size_hw)
    if h <= 0 or w <= 0:
        raise ValueError('Input height and width must be positive')
    if len(mean) != 3 or len(std) != 3 or np.any(np.asarray(std) <= 0):
        raise ValueError('Expected three means and three positive standard deviations')
    # Exactly one BGR -> RGB conversion. Resize to the training shape.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / np.float32(255)
    image = (image - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def decode_output(output, metric_range=(0.2, 1.0)):
    """[1,2,H,W] -> meters and near-head probability, at model resolution."""
    output = np.asarray(output, dtype=np.float32)
    if output.ndim != 4 or output.shape[:2] != (1, 2):
        raise ValueError('Expected model output [1, 2, H, W], got {}'.format(output.shape))
    if not np.isfinite(output).all():
        raise ValueError('Model returned NaN or Inf')
    near, far = (float(v) for v in metric_range)
    if not 0 < near < far:
        raise ValueError('Expected 0 < min_depth_m < max_depth_m')
    unit = np.clip(output[0, 0], 0, 1)
    depth = 1.0 / (1.0 / far + unit * (1.0 / near - 1.0 / far))
    probability = 1.0 / (1.0 + np.exp(-np.clip(output[0, 1], -80, 80)))
    return depth.astype(np.float32), probability.astype(np.float32)


def visualize(frame, depth_m, near_probability, metric_range, mode='side-by-side'):
    """Fixed meter scale across frames; the colors never represent temperature."""
    h, w = frame.shape[:2]
    near, far = metric_range
    gray = np.round(np.clip((far-depth_m)/(far-near), 0, 1)*255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == 'depth':
        return color
    if mode == 'near':
        mask = cv2.resize(near_probability, (w, h), interpolation=cv2.INTER_LINEAR) >= .5
        overlay = frame.copy()
        overlay[mask] = (.5*overlay[mask] + np.array([0, 0, 127.5])).astype(np.uint8)
        return overlay
    if mode != 'side-by-side':
        raise ValueError('Unknown visualization mode: {}'.format(mode))
    return np.concatenate((frame, color), axis=1)
