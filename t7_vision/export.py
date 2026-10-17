"""Export static FP32 ONNX, optionally checking synthetic raw-output parity."""
import argparse
import inspect
import json
from pathlib import Path

from .runtime import load_local_model, sha256_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dualtac-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--verify', action='store_true', help='ONNX Runtime CPU parity (requires onnxruntime)')
    args = p.parse_args(argv)
    import numpy as np
    import torch
    import onnx
    model, contract = load_local_model(args.dualtac_root/'src'/'depth_encoder_decoder.py',
                                       args.dualtac_root/'checkpoints'/'depth_pretrained.pt')
    model.cpu().float()
    torch.set_num_threads(4)
    h, w = contract['input_size_hw']
    dummy = torch.zeros(1, 3, h, w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=['rgb_normalized'], output_names=['depth_and_near_logits'], opset_version=args.opset)
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False
    torch.onnx.export(model, dummy, str(args.output), **kwargs)
    onnx.checker.check_model(onnx.load(str(args.output)))
    verification = {'status': 'not_run', 'scope': 'raw output parity, not depth accuracy or Jetson qualification'}
    if args.verify:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
        rng, errors = np.random.RandomState(7), []
        for x in [np.zeros((1, 3, h, w), np.float32), rng.normal(0, 1, (1, 3, h, w)).astype(np.float32)]:
            with torch.no_grad():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(None, {'rgb_normalized': x})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
            errors.append(float(np.max(np.abs(actual-expected))))
        verification.update({'status': 'passed', 'synthetic_cases': 2, 'max_abs_errors': errors,
                             'rtol': 1e-3, 'atol': 1e-4, 'onnxruntime': ort.__version__})
    manifest = {'schema_version': 1, 'contract': contract, 'onnx_sha256': sha256_file(args.output),
                'opset': args.opset, 'torch': torch.__version__, 'verification': verification}
    Path(str(args.output)+'.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
