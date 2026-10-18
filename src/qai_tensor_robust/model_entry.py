"""Fresh Qwen3 full-model adapters. Asset paths must already be verified locally.

No scheduler, downloads, remote code, or scientific execution at import time.
"""
import json
from pathlib import Path
import torch
from .operators import MatrixFactors, MPO, Layout
from .accounting import deployment_size, deployment_payload, decode_payload

MODEL_REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'


def load_native(model_path, *, device='cpu'):
    from transformers import AutoModelForCausalLM
    path = Path(model_path).resolve(strict=True)
    config = json.loads((path / 'config.json').read_text())
    expected = {'model_type': 'qwen3', 'hidden_size': 1024,
                'intermediate_size': 3072, 'num_hidden_layers': 28}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unexpected model config {key}')
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def replace_down_projections(model, point):
    """Only first-pair O-W/no-repair. FP32 construction, BF16 factors, FP32 scale."""
    c = point['coordinates']
    if (c['C1'], c['C7'], c['C8'], c['C9']) != ('S-D', 'O-W', 'T-None', 'U-None'):
        raise NotImplementedError('Only first-pair O-W construction is implemented')
    layers = model.model.layers
    if len(layers) != 28 or len(point['layer_rank_vector']) != 28:
        raise ValueError('Expected 28 down projections')
    # Validate all targets before changing any layer.
    for layer in layers:
        if tuple(layer.mlp.down_proj.weight.shape) != (1024, 3072):
            raise ValueError('Unexpected down projection shape')
    diagnostics = []
    for index, (layer, rank) in enumerate(zip(layers, point['layer_rank_vector'])):
        old = layer.mlp.down_proj
        weight = old.weight.detach().float()
        scale = torch.ones(3072, device=weight.device, dtype=torch.float32)
        bias = None if old.bias is None else old.bias.detach()
        if c['C2'] == 'F-Matrix':
            operator = MatrixFactors.from_weight(weight, rank, scale=scale, bias=bias)
            detail = {'rank': rank}
        elif c['C2'] == 'F-MPO':
            layout = Layout((8, 8, 16), (12, 16, 16))
            operator, detail = MPO.from_weight(weight, layout, tuple(rank), scale=scale, bias=bias)
        else:
            raise NotImplementedError(c['C2'])
        # Cast factors only; .to(bfloat16) on the whole operator would cast the scale too.
        for parameter in operator.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
            parameter.requires_grad_(False)
        if operator.bias is not None:
            operator.bias = operator.bias.to(torch.bfloat16)
        layer.mlp.down_proj = operator.eval()
        diagnostics.append({'layer': index, 'operator': operator.deployment_metadata(),
                            'construction': detail, 'operator_payload_bytes': deployment_size(operator)})
        del weight, old
    return diagnostics


def operator_from_payload(payload):
    metadata, tensors = decode_payload(payload)
    if metadata['kind'] == 'matrix_factors':
        return MatrixFactors(tensors['left'], tensors['right'],
                             tensors.get('input_scale'), tensors.get('bias')).eval()
    if metadata['kind'] == 'mpo':
        layout = Layout(tuple(metadata['output_modes']), tuple(metadata['input_modes']))
        cores = [tensors[f'cores.{i}'] for i in range(len(layout.input_modes))]
        return MPO(cores, layout, tensors.get('input_scale'), tensors.get('bias')).eval()
    raise ValueError('Unsupported operator payload')


def export_operator(operator, destination):
    """Exact operator payload; never claim this is the complete model export."""
    payload = deployment_payload(operator)
    target = Path(destination)
    with target.open('xb') as stream:
        stream.write(payload)
    return {'bytes': len(payload), 'scope': 'operator_only'}


def load_factorized_model(payload_path, *, device='cpu'):
    """Reload full Qwen3 payload, including attention mode and rotary buffers."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    metadata, tensors = decode_payload(Path(payload_path).read_bytes())
    if metadata['kind'] != 'qwen3_factorized':
        raise ValueError('Not a complete factorized model payload')
    config = Qwen3Config(**metadata['config'])
    config._attn_implementation = metadata['attention_implementation']
    model = Qwen3ForCausalLM(config).eval()
    for prefix, descriptor in metadata['operators'].items():
        if descriptor['kind'] == 'matrix_factors':
            op = MatrixFactors(tensors[prefix+'.left'], tensors[prefix+'.right'],
                               tensors[prefix+'.input_scale'])
        elif descriptor['kind'] == 'mpo':
            layout = Layout(tuple(descriptor['output_modes']), tuple(descriptor['input_modes']))
            op = MPO([tensors[prefix+f'.cores.{i}'] for i in range(len(layout.input_modes))],
                     layout, tensors[prefix+'.input_scale'])
        else:
            raise ValueError('Unsupported full-model operator')
        parent, _, leaf = prefix.rpartition('.')
        setattr(model.get_submodule(parent), leaf, op.eval())
    model.load_state_dict(tensors, strict=True, assign=True)
    model.tie_weights()
    for name, entry in metadata['nonpersistent_buffers'].items():
        parent, _, leaf = name.rpartition('.')
        module = model.get_submodule(parent) if parent else model
        setattr(module, leaf, torch.tensor(entry['values'], dtype=getattr(torch, entry['dtype'].split('.')[-1])))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()
