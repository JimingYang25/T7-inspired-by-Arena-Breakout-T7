"""Camera/file entry point, importable without torch."""
import argparse
from pathlib import Path
import sys
from time import perf_counter


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def parser():
    p = argparse.ArgumentParser(description='T7 near-field depth deployment using a local lightweight DualTac model')
    p.add_argument('--backend', choices=['pytorch', 'onnx'], default='pytorch')
    p.add_argument('--dualtac-root', type=Path, help='Local DualTacV2 folder (not uploaded)')
    p.add_argument('--model-source', type=Path, help='File defining TinyMetricDepthStudent')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--onnx-model', type=Path)
    p.add_argument('--provider', default='CPUExecutionProvider')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--source', default='0', help='Camera index, local image or video')
    p.add_argument('--source-type', choices=['auto', 'camera', 'image', 'video'], default='auto')
    p.add_argument('--width', type=positive, default=640, help='Requested capture width; model size comes from checkpoint')
    p.add_argument('--height', type=positive, default=480)
    p.add_argument('--threads', type=positive, default=4)
    p.add_argument('--warmup', type=nonnegative, default=5)
    p.add_argument('--frames', type=nonnegative, default=0, help='Measured limit; 0 = all video/camera or one image')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--view', choices=['side-by-side', 'depth', 'near'], default='side-by-side')
    p.add_argument('--output', type=Path, help='PNG for images or AVI/MJPG for video')
    p.add_argument('--output-fps', type=float, default=15.0, help='Playback FPS, not measured inference FPS')
    p.add_argument('--save-depth', type=Path, help='Save last model-resolution depth_m and near_probability NPZ')
    p.add_argument('--report', type=Path, default=Path('runs/last.json'))
    return p


def model_paths(args):
    source, checkpoint = args.model_source, args.checkpoint
    if args.dualtac_root:
        source = source or args.dualtac_root/'src'/'depth_encoder_decoder.py'
        checkpoint = checkpoint or args.dualtac_root/'checkpoints'/'depth_pretrained.pt'
    if not source or not checkpoint:
        raise ValueError('Use --dualtac-root PATH or both --model-source FILE --checkpoint FILE')
    return source, checkpoint


class Source:
    def __init__(self, args):
        import cv2
        self.capture, self.image = None, None
        kind = args.source_type
        if kind == 'auto':
            if args.source.isdecimal():
                kind = 'camera'
            elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
                kind = 'image'
            else:
                kind = 'video'
        self.kind = kind
        if kind == 'image':
            self.image = cv2.imread(args.source)
            if self.image is None:
                raise ValueError('Cannot decode image: {}'.format(args.source))
        else:
            target = int(args.source) if kind == 'camera' and args.source.isdecimal() else args.source
            self.capture = cv2.VideoCapture(target)
            if not self.capture.isOpened():
                self.capture.release()
                raise ValueError('Cannot open {}: {}'.format(kind, args.source))
            if kind == 'camera':
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort

    def read(self):
        if self.image is not None:
            return True, self.image.copy()
        return self.capture.read()

    def close(self):
        if self.capture is not None:
            self.capture.release()


def run(args, runtime):
    import cv2
    import numpy as np
    from .processing import decode_output, visualize
    from .metrics import summarize, write_report
    if not np.isfinite(args.output_fps) or args.output_fps <= 0:
        raise ValueError('--output-fps must be finite and positive')
    if args.output and args.output.suffix.lower() not in ['.png', '.avi']:
        raise ValueError('--output must end in .png (image) or .avi (video)')
    source, writer, samples = None, None, []
    status, error, depth_m, probability = 'complete', None, None, None
    exclude = ['checkpoint', 'model_source', 'dualtac_root', 'onnx_model', 'report', 'source', 'output', 'save_depth']
    config = {k: v for k, v in vars(args).items() if k not in exclude}
    config['source_name'] = Path(args.source).name
    config['outputs'] = {'visualization': bool(args.output), 'last_depth_npz': bool(args.save_depth)}
    try:
        source = Source(args)
        config['source_type'] = source.kind
        if args.output and (source.kind == 'image') != (args.output.suffix.lower() == '.png'):
            raise ValueError('Use PNG for an image and AVI for video/camera')
        # Warmup uses synthetic input at the exact network shape, consuming no source frames.
        h, w = runtime.contract['input_size_hw']
        blank = np.zeros((h, w, 3), np.uint8)
        for _ in range(args.warmup):
            runtime.predict(blank)
        limit = args.frames or (1 if source.kind == 'image' else 0)
        print('Backend: {} | input: {} | range: {} m'.format(runtime.info['backend'], [h, w], runtime.contract['metric_range_m']))
        while not limit or len(samples) < limit:
            start = perf_counter()
            ok, frame = source.read()
            read_end = perf_counter()
            if not ok:
                if source.kind == 'camera' or not samples:
                    raise RuntimeError('Capture failed before a usable frame')
                break
            if not samples:
                config['actual_frame_size_hw'] = list(frame.shape[:2])
            output, timing = runtime.predict(frame)
            post_start = perf_counter()
            depth_m, probability = decode_output(output, runtime.contract['metric_range_m'])
            rendered = None
            if not args.headless or args.output:
                rendered = visualize(frame, depth_m, probability, runtime.contract['metric_range_m'], args.view)
                cv2.putText(rendered, 'Depth estimate | range {:.2f}-{:.2f} m'.format(*runtime.contract['metric_range_m']),
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            post_end = perf_counter()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if source.kind == 'image':
                    if not cv2.imwrite(str(args.output), rendered):
                        raise RuntimeError('Failed to write output image')
                else:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'MJPG'), args.output_fps,
                                                 (rendered.shape[1], rendered.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot initialize MJPG AVI writer')
                    writer.write(rendered)
            stop = False
            if not args.headless:
                cv2.imshow('T7 near-field depth', rendered)
                stop = cv2.waitKey(1) & 0xff in (27, ord('q'))
            end = perf_counter()
            timing.update({'read_ms': (read_end-start)*1000, 'postprocess_ms': (post_end-post_start)*1000,
                           'output_ui_ms': (end-post_end)*1000, 'pipeline_ms': (end-start)*1000})
            samples.append(timing)
            if len(samples) % 30 == 0:
                print('Frames {} | pipeline {:.2f} FPS | prediction {:.2f} ms'.format(
                    len(samples), len(samples)*1000/sum(s['pipeline_ms'] for s in samples), timing['predict_ms']))
            if stop:
                status = 'stopped_by_user'
                break
        if args.save_depth is not None and depth_m is not None:
            args.save_depth.parent.mkdir(parents=True, exist_ok=True)
            with args.save_depth.open('wb') as stream:
                np.savez_compressed(stream, depth_m=depth_m, near_probability=probability)
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        status, error = 'failed', str(exc)
        raise
    finally:
        if writer is not None:
            writer.release()
        if source is not None:
            source.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # A headless OpenCV build may fail before any window exists.
        report = summarize(samples, runtime, config, status, error)
        write_report(args.report, report)
        print('Report: {} | measured frames: {}'.format(args.report, len(samples)))
    return 130 if status == 'interrupted' else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .runtime import TorchRuntime, OnnxRuntime
        if args.backend == 'pytorch':
            source, checkpoint = model_paths(args)
            runtime = TorchRuntime(source, checkpoint, args.device, args.precision, args.threads)
        else:
            if not args.onnx_model:
                raise ValueError('--backend onnx requires --onnx-model')
            if args.device != 'auto' or args.precision != 'fp32':
                raise ValueError('For ONNX select --provider; --device/--precision apply to PyTorch')
            runtime = OnnxRuntime(args.onnx_model, args.provider, args.threads)
        return run(args, runtime)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print('Error: {}'.format(exc), file=sys.stderr)
        return 1
