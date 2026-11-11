"""Export the selected TrackNet heatmap in float32 for local CPU inference."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from head_adapter import model_settings
from neural_candidates import ROOT, WEIGHTS, load_tracknet
from onnx_ball import FORMAT, bundle_path


class Heatmap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        return self.model(frames=frames)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--head-adapter', type=Path)
    selection.add_argument('--base-model', action='store_true')
    args = parser.parse_args()
    adapter, _, _, _ = model_settings(args.head_adapter, args.base_model)
    weights_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest() if adapter else None
    graph = bundle_path(ROOT, weights_hash, adapter_hash)
    graph.parent.mkdir(parents=True, exist_ok=True)
    pending_graph = graph.with_suffix('.pending.onnx')
    model = Heatmap(load_tracknet(weights_hash, adapter)).eval()
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros((1, 15, 144, 256), dtype=torch.float32), str(pending_graph),
                          dynamo=False, opset_version=17, input_names=['frames'], output_names=['heatmap'],
                          dynamic_axes={'frames':{2:'height', 3:'width'}, 'heatmap':{2:'height', 3:'width'}})
    manifest = {'format':FORMAT, 'weightsSha256':weights_hash, 'headAdapterSha256':adapter_hash,
                'onnxSha256':hashlib.sha256(pending_graph.read_bytes()).hexdigest()}
    pending_graph.replace(graph)
    graph.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(graph)


if __name__ == '__main__':
    main()
