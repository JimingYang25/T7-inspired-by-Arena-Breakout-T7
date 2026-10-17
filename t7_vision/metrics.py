"""Measured distributions and machine-readable provenance."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered)-1)*q
    low, high = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[low] + (ordered[high]-ordered[low])*(pos-low)


def summarize(samples, runtime, config, status='complete', error=None):
    distributions = {}
    if samples:
        for key in samples[0]:
            values = [s[key] for s in samples]
            distributions[key] = {'mean': sum(values)/len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95)}
    total_ms = sum(s['pipeline_ms'] for s in samples)
    return {'schema_version': 1, 'status': status, 'error': error,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'host': {'os': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
            'runtime': runtime.info, 'config': config, 'measured_frames': len(samples),
            'pipeline_fps': len(samples)*1000/total_ms if total_ms else None,
            'timings_ms': distributions, 'memory': runtime.memory(),
            'scope': 'Read + preprocessing + synchronized inference + transfer + decoding + requested visualization/output/UI. Excludes initialization, warmup, report writing and final NPZ saving. Not sensor-to-display timing.',
            'samples': samples}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)
