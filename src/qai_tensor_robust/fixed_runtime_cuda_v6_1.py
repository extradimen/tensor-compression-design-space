"""Production CUDA workload adapter, invoked only inside the protected .4 child.

Existing NVIDIA driver telemetry is read-only. No model/data download, scheduler,
GPU configuration change, billing estimate or power-as-process attribution.
"""
import copy
import os
import subprocess
import threading
import time

from .fixed_model_measurement_v6 import integrate_board_energy, measure_fixed_workload
from .runtime_components_v6_1 import summarize_forward_components


class BoardTelemetry:
    def __init__(self, uuid):
        if not uuid.startswith('GPU-') or uuid != os.environ.get('QAI_GPU_UUIDS'):
            raise ValueError('Only the assigned protected UUID may be sampled')
        self.uuid, self.samples, self.error = uuid, [], None
        self.stop = threading.Event()
        self.ready = threading.Event()

    def _sample(self):
        try:
            while not self.stop.is_set():
                before = time.monotonic()
                value = subprocess.run(['nvidia-smi', '-i', self.uuid,
                    '--query-gpu=power.draw,temperature.gpu,memory.used',
                    '--format=csv,noheader,nounits'], capture_output=True,
                    text=True, timeout=2, check=True).stdout.strip()
                after = time.monotonic()
                watts, temperature, used = (float(x.strip()) for x in value.split(','))
                self.samples.append({'monotonic_seconds': (before+after)/2,
                    'watts':watts, 'temperature_C':temperature,
                    'board_memory_used_bytes':used*1024*1024,
                    'query_wall_seconds':after-before})
                self.ready.set()
                self.stop.wait(.1)
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            self.ready.set()

    def __enter__(self):
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        if not self.ready.wait(3) or self.error or not self.samples:
            self.__exit__(None,None,None)
            raise ValueError('Existing GPU power telemetry unavailable: '+str(self.error))
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.worker.join(3)
        if self.worker.is_alive():
            raise RuntimeError('Telemetry subprocess/thread cleanup unconfirmed')

    def cover(self, end):
        deadline = time.monotonic()+3
        while self.samples[-1]['monotonic_seconds'] < end:
            if self.error or time.monotonic() > deadline:
                raise ValueError('Task end not covered by GPU power telemetry')
            self.stop.wait(.02)


def measure_fixed_model(model, records, uuid, *, on_task_completed=None):
    """Return NLL-workload and fixed-decode reports, with raw board samples.

    Caller must free construction/optimizer/teacher GPU state before calling.
    This adapter cannot certify arbitrary GPU tensors held elsewhere by caller.
    """
    import torch
    from .task_loss import causal_task_loss
    if (not os.environ.get('QAI_ATTEMPT_ID') or torch.cuda.device_count()!=1
            or model.training or any(p.requires_grad for p in model.parameters())
            or any(p.device.type!='cuda' for p in model.parameters())):
        raise ValueError('Protected loaded/frozen eval CUDA model required')
    if len(records)!=36 or {r['domain'] for r in records}!={'text','math'}:
        raise ValueError('Exact Search36 task required')
    device = next(model.parameters()).device
    prepared=[]
    for r in records:
        ids=torch.tensor([r['input_ids']],dtype=torch.long,device=device)
        mask=torch.tensor([r['attention_mask']],dtype=torch.bool,device=device)
        targets=torch.tensor([r['target_mask']],dtype=torch.bool,device=device)
        if ids.shape!=(1,512) or mask.shape!=ids.shape or targets.shape!=ids.shape:
            raise ValueError('Fixed first512 workload shape differs')
        prepared.append((r['domain'],ids,mask,targets))
    decode=[next(x for x in prepared if x[0]==domain) for domain in ('text','math')]
    generation=copy.deepcopy(model.generation_config)
    generation.do_sample=False
    generation.num_beams=1
    generation.max_new_tokens=32
    generation.min_new_tokens=32
    generation.eos_token_id=None
    generation.use_cache=True
    forward_event_history=[]
    def nll_work():
        with torch.inference_mode():
            for _,ids,mask,targets in prepared:
                out=model(ids,attention_mask=mask,use_cache=False)
                loss=causal_task_loss(out.logits,ids,targets)
                if not torch.isfinite(loss.ce_sum):
                    raise ValueError('Nonfinite NLL during runtime task')
                del out,loss
        return {'input_tokens':sum(int(m.sum()) for _,_,m,_ in prepared),
                'scored_tokens':sum(int(t[:,1:].sum()) for _,_,_,t in prepared)}
    def decode_work():
        events=[]
        active=[]
        def before_forward(module,args,kwargs):
            ids=kwargs.get('input_ids')
            if ids is None and args:ids=args[0]
            if ids is None or ids.ndim!=2 or ids.shape[0]!=1:
                raise ValueError('Unexpected generated-forward input')
            begin=torch.cuda.Event(enable_timing=True)
            begin.record()
            active.append((begin,int(ids.shape[1])))
        def after_forward(module,args,kwargs,output):
            begin,length=active.pop()
            end=torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((begin,end,length))
        pre=model.register_forward_pre_hook(before_forward,with_kwargs=True)
        post=model.register_forward_hook(after_forward,with_kwargs=True)
        try:
            with torch.inference_mode():
                for _,ids,mask,_ in decode:
                    result=model.generate(ids,attention_mask=mask,generation_config=generation)
                    if result.shape!=(1,544):
                        raise ValueError('Fixed decode work count changed')
                    del result
            if active:raise ValueError('Unclosed generation forward event')
            forward_event_history.append(events)
        finally:
            pre.remove();post.remove()
        return {'input_tokens':sum(int(m.sum()) for _,_,m,_ in decode),
                'generated_tokens':64,'prompt_slots':1024}
    def memory():
        return {'cuda_allocated':torch.cuda.max_memory_allocated(),
                'cuda_reserved':torch.cuda.max_memory_reserved()}
    reports={}
    for name,work in [('search_nll',nll_work),('fixed_decode',decode_work)]:
        with BoardTelemetry(uuid) as meter:
            report=measure_fixed_workload(work,torch.cuda.synchronize,
                torch.cuda.reset_peak_memory_stats,memory,clock=time.monotonic)
            meter.cover(report['records'][-1]['end_monotonic_seconds'])
            for row in report['records']:
                row.update(integrate_board_energy(meter.samples,
                    row['start_monotonic_seconds'],row['end_monotonic_seconds']))
                row['energy_status']='MEASURED_BOARD_ENERGY'
            report['raw_board_telemetry']=list(meter.samples)
            report['power_timestamp_method']='host midpoint of each read-only driver query; query latency retained'
            report['GPU_UUID']=uuid
            report['scientific_GPU_measurement_claimed']=True
            if name=='fixed_decode':
                if len(forward_event_history)!=7:
                    raise ValueError('Expected2 warmups plus5 timed generation histories')
                for row,events in zip(report['records'],forward_event_history[2:]):
                    row['forward_components']=summarize_forward_components([
                        {'input_length':length,'cuda_elapsed_ms':begin.elapsed_time(end)}
                        for begin,end,length in events])
                report['prefill_decode_components_separately_measured']=True
                report['generation_event_instrumentation_applied_to_all_models']=True
            else:
                report['prefill_decode_components_separately_measured']='NOT_APPLICABLE_TO_NLL_WORKLOAD'
            reports[name]=report
            if on_task_completed is not None:
                on_task_completed(name,report)
    return reports
