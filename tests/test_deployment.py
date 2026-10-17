import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from t7_vision.cli import parser, run, model_paths
from t7_vision.metrics import summarize
from t7_vision.processing import preprocess_bgr, decode_output, visualize, MEAN, STD


class FakeRuntime:
    def __init__(self):
        self.calls = 0
        self.contract = {'input_size_hw': [24, 32], 'metric_range_m': [.2, 1.]}
        self.info = {'backend': 'test_fixture', 'contract': self.contract}

    def predict(self, frame):
        self.calls += 1
        output = np.zeros((1, 2, 24, 32), np.float32)
        output[:, 0] = .5
        return output, {'preprocess_ms': .1, 'forward_ms': .2, 'transfer_ms': 0., 'predict_ms': .3}

    def memory(self):
        return None


class ProcessingTests(unittest.TestCase):
    def test_rgb_order_normalization_and_layout(self):
        image = np.full((2, 3, 3), [0, 128, 255], dtype=np.uint8)
        x = preprocess_bgr(image, (2, 3))
        expected = (np.array([1, 128/255, 0], np.float32)-MEAN)/STD
        np.testing.assert_allclose(x[0, :, 0, 0], expected, atol=1e-6)
        self.assertEqual(x.shape, (1, 3, 2, 3))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.flags.c_contiguous)

    def test_rejects_empty_or_wrong_channels(self):
        for image in [np.zeros((0, 1, 3), np.uint8), np.zeros((2, 3), np.uint8), np.zeros((2, 3, 3), np.float32)]:
            with self.assertRaises(ValueError):
                preprocess_bgr(image)

    def test_fixed_inverse_depth_decoding(self):
        raw = np.zeros((1, 2, 1, 3), np.float32)
        raw[0, 0] = [0, .5, 1]
        meters, near = decode_output(raw)
        np.testing.assert_allclose(meters, [[1., 1/3, .2]], rtol=1e-6)
        np.testing.assert_equal(near, .5)

    def test_invalid_output_is_not_rendered(self):
        for raw in [np.zeros((1, 1, 2, 3)), np.full((1, 2, 2, 3), np.nan)]:
            with self.assertRaises(ValueError):
                decode_output(raw)

    def test_extreme_logits_are_finite(self):
        raw = np.zeros((1, 2, 1, 2), np.float32)
        raw[0, 1] = [-1e8, 1e8]
        _, p = decode_output(raw)
        self.assertTrue(np.isfinite(p).all())
        self.assertLess(p[0, 0], 1e-20)
        self.assertEqual(p[0, 1], 1)

    def test_three_channel_visualization_and_fixed_scale(self):
        frame = np.zeros((40, 60, 3), np.uint8)
        meters = np.full((10, 15), .4, np.float32)
        probability = np.zeros_like(meters)
        side = visualize(frame, meters, probability, (.2, 1.))
        depth = visualize(frame, meters, probability, (.2, 1.), 'depth')
        self.assertEqual(side.shape, (40, 120, 3))
        self.assertEqual(side.dtype, np.uint8)
        np.testing.assert_equal(side[:, 60:], depth)
        self.assertEqual(visualize(frame, meters, probability, (.2, 1.), 'near').shape, frame.shape)


class PipelineTests(unittest.TestCase):
    def test_video_eof_warmup_and_frame_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/'fixture.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(8):
                writer.write(np.full((48, 64, 3), 20*i, np.uint8))
            writer.release()
            args = parser().parse_args(['--source', str(video), '--headless', '--warmup', '3',
                                       '--report', str(root/'report.json'), '--output', str(root/'out.avi')])
            runtime = FakeRuntime()
            self.assertEqual(run(args, runtime), 0)
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['measured_frames'], 8)
            self.assertEqual(runtime.calls, 11)
            self.assertEqual(report['config']['actual_frame_size_hw'], [48, 64])
            cap = cv2.VideoCapture(str(root/'out.avi'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 128)
            cap.release()
            args.frames, args.output = 2, None
            self.assertEqual(run(args, FakeRuntime()), 0)
            self.assertEqual(json.loads((root/'report.json').read_text())['measured_frames'], 2)

    def test_image_saves_visualization_and_numeric_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root/'input.png'), np.zeros((48, 64, 3), np.uint8))
            args = parser().parse_args(['--source', str(root/'input.png'), '--headless', '--warmup', '0',
                                       '--output', str(root/'out.png'), '--save-depth', str(root/'depth.npz'),
                                       '--report', str(root/'report.json')])
            self.assertEqual(run(args, FakeRuntime()), 0)
            self.assertEqual(cv2.imread(str(root/'out.png')).shape, (48, 128, 3))
            with np.load(str(root/'depth.npz')) as data:
                self.assertEqual(data['depth_m'].shape, (24, 32))
                np.testing.assert_allclose(data['depth_m'], 1/3)

    def test_bad_source_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(['--source', str(root/'missing.png'), '--headless', '--report', str(root/'report.json')])
            with self.assertRaises(ValueError):
                run(args, FakeRuntime())
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['measured_frames'], 0)
            self.assertIsNone(report['pipeline_fps'])

    def test_invalid_numeric_arguments(self):
        for pair in [['--warmup', '-1'], ['--threads', '0'], ['--frames', '-5']]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args(pair)

    def test_model_paths_have_no_machine_specific_default(self):
        with self.assertRaises(ValueError):
            model_paths(parser().parse_args([]))

    def test_report_percentiles_and_pipeline_fps(self):
        samples = [{'pipeline_ms': 10., 'forward_ms': 2.}, {'pipeline_ms': 30., 'forward_ms': 6.}]
        report = summarize(samples, FakeRuntime(), {})
        self.assertEqual(report['pipeline_fps'], 50.)
        self.assertEqual(report['timings_ms']['forward_ms']['p50'], 4.)


@unittest.skipUnless(os.environ.get('T7_DUALTAC_ROOT'), 'private checkpoint integration is opt-in')
class LocalModelTests(unittest.TestCase):
    def test_real_checkpoint_cpu_forward_and_immutability(self):
        from t7_vision.runtime import TorchRuntime, sha256_file
        root = Path(os.environ['T7_DUALTAC_ROOT'])
        ckpt = root/'checkpoints'/'depth_pretrained.pt'
        before = sha256_file(ckpt)
        runtime = TorchRuntime(root/'src'/'depth_encoder_decoder.py', ckpt, 'cpu')
        raw, _ = runtime.predict(np.full((480, 640, 3), [0, 128, 255], np.uint8))
        self.assertEqual(runtime.contract['parameters'], 937227)
        self.assertEqual(raw.shape, (1, 2, 240, 320))
        meters, p = decode_output(raw)
        self.assertTrue(np.isfinite(meters).all())
        self.assertTrue(((meters >= .2) & (meters <= 1.)).all())
        self.assertEqual(before, sha256_file(ckpt))


@unittest.skipUnless(os.environ.get('T7_ONNX_MODEL'), 'exported private ONNX integration is opt-in')
class OnnxTests(unittest.TestCase):
    def test_end_to_end_pytorch_onnx_input_output_parity(self):
        from t7_vision.runtime import OnnxRuntime, TorchRuntime
        root = Path(os.environ['T7_DUALTAC_ROOT'])
        image = np.full((480, 640, 3), [17, 128, 255], np.uint8)
        pt = TorchRuntime(root/'src'/'depth_encoder_decoder.py', root/'checkpoints'/'depth_pretrained.pt', 'cpu')
        ort = OnnxRuntime(os.environ['T7_ONNX_MODEL'])
        expected, _ = pt.predict(image)
        actual, _ = ort.predict(image)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        dm_pt, near_pt = decode_output(expected)
        dm_ort, near_ort = decode_output(actual)
        np.testing.assert_allclose(dm_ort, dm_pt, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(near_ort, near_pt, rtol=1e-3, atol=1e-4)

    def test_tampered_contract_rejected(self):
        import shutil
        from t7_vision.runtime import OnnxRuntime
        path = Path(os.environ['T7_ONNX_MODEL'])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'model.onnx'
            shutil.copyfile(str(path), str(destination))
            manifest = json.loads(Path(str(path)+'.json').read_text())
            manifest['onnx_sha256'] = '0'*64
            Path(str(destination)+'.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'hash'):
                OnnxRuntime(destination)


if __name__ == '__main__':
    unittest.main()
