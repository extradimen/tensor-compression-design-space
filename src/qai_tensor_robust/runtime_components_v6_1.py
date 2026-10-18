"""Pure validation/reduction of fixed-workload CUDA forward event records."""
import math


def summarize_forward_components(events, prompt_count=2, generated_per_prompt=32):
    if type(prompt_count) is not int or prompt_count<1 or generated_per_prompt!=32:
        raise ValueError('Fixed prompt/token counts required')
    if len(events)!=prompt_count*generated_per_prompt:
        raise ValueError('Generation forward count differs from fixed workload')
    prefill,decode=0.,0.
    for prompt in range(prompt_count):
        block=events[prompt*32:(prompt+1)*32]
        if block[0]['input_length']!=512 or any(e['input_length']!=1 for e in block[1:]):
            raise ValueError('Generation must be one prefill plus31 cached decode forwards')
        for i,e in enumerate(block):
            ms=e['cuda_elapsed_ms']
            if isinstance(ms,bool) or not isinstance(ms,(int,float)) or not math.isfinite(ms) or ms<0:
                raise ValueError('Invalid CUDA event time')
            if i==0:prefill+=ms/1000
            else:decode+=ms/1000
    return {'prefill_forward_cuda_seconds':prefill,'decode_forward_cuda_seconds':decode,
            'prefill_forward_calls':prompt_count,'decode_forward_calls':prompt_count*31,
            'first_tokens_from_prefill':prompt_count,'tokens_from_decode_forwards':prompt_count*31,
            'generated_tokens':prompt_count*32,
            'component_scope':'CUDA event intervals around model forwards; total synchronized host task time reported separately',
            'component_energy_individually_measured':False}


def validate_worker_uuid(config_uuid, ticket_uuid, visible_count, compute_rows, current_pid):
    """Bind CUDA worker process to its exact assigned physical board."""
    if (not isinstance(config_uuid,str) or not config_uuid.startswith('GPU-')
            or config_uuid!=ticket_uuid or ',' in ticket_uuid or visible_count!=1):
        raise ValueError('One exact config/controller-ticket UUID required')
    owned=[r for r in compute_rows if r['pid']==current_pid]
    if len(owned)!=1 or owned[0]['gpu_uuid']!=ticket_uuid:
        raise ValueError('Actual CUDA process does not map uniquely to ticket UUID')
    if any(r['gpu_uuid']==ticket_uuid and r['pid']!=current_pid for r in compute_rows):
        raise ValueError('Other compute process on benchmark board')
    return {'actual_process_pid':current_pid,'actual_physical_GPU_UUID':ticket_uuid,
            'torch_visible_count':visible_count,'torch_device':'cuda:0','other_board_compute_processes':0}
