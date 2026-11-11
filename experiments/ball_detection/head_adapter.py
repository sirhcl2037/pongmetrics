"""Apply a small, checkpoint-bound final-layer adapter without changing the backbone."""
import hashlib
import json
from pathlib import Path

import numpy as np


def model_settings(adapter=None, base_model=False, strong_score=None, acquisition_score=None, default_path=None):
    if adapter is not None and base_model:
        raise ValueError('Choose an adapter or the base model')
    default = Path(default_path) if default_path is not None else Path(__file__).parent/'adapters/review-head-v1.json'
    selected = None
    if not base_model:
        if adapter is not None:
            selected = Path(adapter)
        elif default.exists():
            selected = default
    metadata = json.loads(selected.read_text(encoding='utf-8')) if selected else {}
    if not isinstance(metadata,dict) or not isinstance(metadata.get('tracking',{}),dict):
        raise ValueError('Invalid adapter metadata')
    tracking = metadata.get('tracking', {})
    association = tracking.get('associationScore', .5) if strong_score is None else strong_score
    acquisition = tracking.get('acquisitionScore', association) if acquisition_score is None else acquisition_score
    if not isinstance(association,(int,float)) or not isinstance(acquisition,(int,float)) or not 0 < association <= acquisition <= 1:
        raise ValueError('Tracking scores must satisfy 0 < association <= acquisition <= 1')
    return selected, association, acquisition, metadata.get('name')


def residual_head(original, hidden_weight, hidden_bias, output_weight, output_bias, cap):
    import torch

    class ResidualHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = original
            hidden, channels = hidden_weight.shape
            self.hidden = torch.nn.Conv2d(channels, hidden, 1)
            self.output = torch.nn.Conv2d(hidden, 1, 1)
            self.cap = cap
            with torch.no_grad():
                self.hidden.weight.copy_(torch.tensor(hidden_weight[:,:,None,None]))
                self.hidden.bias.copy_(torch.tensor(hidden_bias))
                self.output.weight.copy_(torch.tensor(output_weight[None,:,None,None]))
                self.output.bias.fill_(output_bias)

        def forward(self, features):
            original_logits = self.original(features)
            residual = self.cap * torch.tanh(self.output(torch.relu(self.hidden(features))) / self.cap)
            return torch.cat([original_logits[:,:-1], original_logits[:,-1:] + residual], dim=1)

    return ResidualHead().eval()


def apply_head_adapter(model, path, base_hash):
    import torch

    raw = Path(path).read_bytes()
    adapter = json.loads(raw)
    channels = model.predictor.weight.shape[1]
    values = np.asarray(adapter.get('weights'), dtype=np.float32)
    if (adapter.get('format') not in ['pongmetrics-linear-head-adapter','pongmetrics-residual-head-adapter'] or adapter.get('version') != 1 or
            adapter.get('baseWeightsSha256') != base_hash or values.shape != (channels,) or
            not np.isfinite(values).all() or not isinstance(adapter.get('bias'), (int, float)) or
            not np.isfinite(adapter['bias'])):
        raise ValueError('Head adapter must match the original checkpoint and predictor shape')
    residual = None
    if adapter['format'] == 'pongmetrics-residual-head-adapter':
        hidden = np.asarray(adapter.get('hiddenWeights'),dtype=np.float32)
        bias = np.asarray(adapter.get('hiddenBias'),dtype=np.float32)
        output = np.asarray(adapter.get('outputWeights'),dtype=np.float32)
        cap = adapter.get('residualCap')
        if (hidden.ndim != 2 or hidden.shape[1] != channels or not 1 <= hidden.shape[0] <= 16 or
                bias.shape != (hidden.shape[0],) or output.shape != bias.shape or
                not all(np.isfinite(value).all() for value in [hidden,bias,output]) or
                not isinstance(adapter.get('outputBias'),(int,float)) or not np.isfinite(adapter['outputBias']) or
                not isinstance(cap,(int,float)) or not 0 < cap <= 4):
            raise ValueError('Invalid residual head shape or parameters')
        residual = (hidden,bias,output,adapter['outputBias'],cap)
    with torch.no_grad():
        model.predictor.weight[-1, :, 0, 0].copy_(torch.tensor(values))
        model.predictor.bias[-1].fill_(adapter['bias'])
    if residual is not None:
        model.predictor = residual_head(model.predictor, *residual)
    return hashlib.sha256(raw).hexdigest()
