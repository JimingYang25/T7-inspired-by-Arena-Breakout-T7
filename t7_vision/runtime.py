"""Private local checkpoint adapter and portable ONNX runtime."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from .processing import MEAN, STD, preprocess_bgr


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_model(source, checkpoint):
    import torch
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Supply local depth source and checkpoint; see --dualtac-root')
    name = '_t7_local_depth_' + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    payload = torch.load(str(checkpoint), **kwargs)
    if not isinstance(payload, dict) or 'model_state_dict' not in payload:
        raise ValueError('Expected clean DualTac checkpoint with model_state_dict')
    config, metadata = payload.get('model_config', {}), payload.get('metadata', {})
    if metadata.get('architecture') != 'dualtac_depth_model':
        raise ValueError('Checkpoint architecture must be dualtac_depth_model')
    model = module.TinyMetricDepthStudent(**config)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if int(metadata.get('parameters', -1)) != parameters:
        raise ValueError('Checkpoint parameter metadata does not match architecture')
    size, norm = metadata.get('training_input_size_hw'), metadata.get('normalization', {})
    limits = metadata.get('metric_range_m')
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not limits or len(limits) != 2:
        raise ValueError('Checkpoint is missing input size or metric depth range')
    if metadata.get('input_layout') != 'NCHW' or metadata.get('input_channels') != 3:
        raise ValueError('Expected NCHW RGB checkpoint')
    preprocess_bgr(np.zeros((2, 2, 3), np.uint8), size, norm.get('mean', MEAN), norm.get('std', STD))
    if not 0 < float(limits[0]) < float(limits[1]):
        raise ValueError('Invalid checkpoint depth range')
    contract = {'architecture': metadata['architecture'], 'parameters': parameters,
                'input_size_hw': list(size), 'mean': norm.get('mean', list(MEAN)),
                'std': norm.get('std', list(STD)), 'metric_range_m': list(limits),
                'near_threshold_m': metadata.get('near_threshold_m', .4),
                'checkpoint_sha256': sha256_file(checkpoint), 'model_source_sha256': sha256_file(source),
                'output_layout': 'NCHW', 'output_channels': ['inverse_depth_unit', 'near_logits']}
    return model.eval(), contract


class TorchRuntime:
    def __init__(self, source, checkpoint, device='auto', precision='fp32', threads=4):
        import torch
        self.torch = torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; check the installed PyTorch build')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('FP16 requires CUDA; use FP32 on CPU')
        torch.set_num_threads(threads)
        self.model, self.contract = load_local_model(source, checkpoint)
        self.device, self.precision = device, precision
        self.model.to(device)
        if precision == 'fp16':
            self.model.half()
        self.info = {'backend': 'pytorch', 'device': device, 'precision': precision,
                     'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                     'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                     'cpu_threads': threads, 'contract': self.contract}
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def predict(self, frame):
        self.sync()
        t0 = perf_counter()
        x = preprocess_bgr(frame, self.contract['input_size_hw'], self.contract['mean'], self.contract['std'])
        x = self.torch.from_numpy(x).to(self.device)
        if self.precision == 'fp16':
            x = x.half()
        self.sync()
        t1 = perf_counter()
        with self.torch.no_grad():
            output = self.model(x)
        self.sync()
        t2 = perf_counter()
        output = output.float().cpu().numpy()
        self.sync()
        t3 = perf_counter()
        return output, {'preprocess_ms': (t1-t0)*1000, 'forward_ms': (t2-t1)*1000,
                        'transfer_ms': (t3-t2)*1000, 'predict_ms': (t3-t0)*1000}

    def memory(self):
        if self.device != 'cuda':
            return None
        return {'cuda_peak_allocated_bytes': self.torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': self.torch.cuda.max_memory_reserved()}


class OnnxRuntime:
    def __init__(self, model_path, provider='CPUExecutionProvider', threads=4):
        import onnxruntime as ort
        path = Path(model_path).resolve()
        sidecar = Path(str(path)+'.json')
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError('ONNX requires its .onnx.json contract from t7_vision.export')
        manifest = json.loads(sidecar.read_text(encoding='utf-8'))
        if manifest.get('onnx_sha256') != sha256_file(path):
            raise ValueError('ONNX hash does not match sidecar; re-export the pair')
        self.contract = manifest['contract']
        if provider not in ort.get_available_providers():
            raise ValueError('Provider {} unavailable; installed: {}'.format(provider, ort.get_available_providers()))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
        inputs = self.session.get_inputs()
        expected = [1, 3] + self.contract['input_size_hw']
        if len(inputs) != 1 or inputs[0].shape != expected or inputs[0].type != 'tensor(float)':
            raise ValueError('ONNX input must be static float32 {}'.format(expected))
        self.input_name = inputs[0].name
        self.info = {'backend': 'onnxruntime', 'provider': provider,
                     'session_providers': self.session.get_providers(), 'precision': 'fp32',
                     'onnxruntime': ort.__version__, 'cpu_threads': threads,
                     'contract': self.contract, 'onnx_sha256': manifest['onnx_sha256']}

    def predict(self, frame):
        t0 = perf_counter()
        x = preprocess_bgr(frame, self.contract['input_size_hw'], self.contract['mean'], self.contract['std'])
        t1 = perf_counter()
        output = self.session.run(None, {self.input_name: x})[0]
        t2 = perf_counter()
        return output, {'preprocess_ms': (t1-t0)*1000, 'forward_ms': (t2-t1)*1000,
                        'transfer_ms': 0.0, 'predict_ms': (t2-t0)*1000}

    def memory(self):
        return None
