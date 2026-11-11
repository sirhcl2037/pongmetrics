"""Local float32 ONNX bundle for an exact checkpoint and head adapter."""
import hashlib
import json
from pathlib import Path


FORMAT = 'pongmetrics-tracknet-onnx-v1'


def bundle_path(root, weights_hash, adapter_hash):
    adapter_key = adapter_hash[:16] if adapter_hash else 'base'
    return Path(root) / 'output' / 'models' / f'tracknet-{weights_hash[:16]}-{adapter_key}.onnx'


def load_session(graph, weights_hash, adapter_hash, threads, require_preference):
    import onnxruntime as ort

    manifest = json.loads(graph.with_suffix('.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or manifest.get('format') != FORMAT or
            manifest.get('weightsSha256') != weights_hash or
            manifest.get('headAdapterSha256') != adapter_hash):
        raise ValueError('ONNX export does not match the selected ball model')
    if require_preference and (manifest.get('preferredRuntime') != ort.__version__ or
                               manifest.get('preferredThreads') != threads):
        return None
    if hashlib.sha256(graph.read_bytes()).hexdigest() != manifest.get('onnxSha256'):
        raise ValueError('ONNX graph does not match its manifest')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    try:
        return ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    except Exception as error:
        raise ValueError(f'ONNX Runtime could not open the ball model: {error}') from error
