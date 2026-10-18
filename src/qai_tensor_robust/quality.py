"""Gold NLL with explicit once-only targets and preaggregated unit identities."""
from dataclasses import dataclass

import torch

from .task_loss import causal_task_loss


@dataclass(frozen=True)
class ScoringWindow:
    start: int
    end: int
    target_start: int
    target_end: int

    def mask(self):
        return [self.target_start <= p < self.target_end for p in range(self.start, self.end)]


def scoring_windows(length, *, window=2048, stride=512):
    if type(length) is not int or length < 2 or not 1 <= stride < window:
        raise ValueError("Need at least one gold target and overlapping context windows")
    end, previous = min(length, window), 1
    windows = []
    while True:
        start = max(0, end - window)
        if previous <= start:
            raise ValueError("A scored target cannot be the first context position")
        windows.append(ScoringWindow(start, end, previous, end))
        if end == length:
            break
        previous, end = end, min(length, end + stride)
    return windows


def text_gold_nll(model, article_ids, *, unit_id, device, window=2048, stride=512):
    """One article produces one bootstrap unit, regardless of window count."""
    total, count = 0., 0
    for w in scoring_windows(len(article_ids), window=window, stride=stride):
        ids = torch.tensor([article_ids[w.start:w.end]], device=device, dtype=torch.long)
        targets = torch.tensor([w.mask()], device=device, dtype=torch.bool)
        with torch.no_grad():
            logits = model(ids, attention_mask=torch.ones_like(targets), use_cache=False).logits
            loss = causal_task_loss(logits, ids, targets)
        total += float(loss.ce_sum)
        count += loss.token_count
    if count != len(article_ids) - 1:
        raise ValueError("Gold text targets were repeated or omitted")
    return {"unit_id": unit_id, "unit_type": "article", "nll_sum": total,
            "target_count": count, "window": window, "stride": stride,
            "metric_scope": "confirmation_2048" if window == 2048 else "pilot_512_different_metric"}


def answer_target_mask(offsets, answer_start, attention_mask):
    """Masks for native raw prompt+reference text tokenized jointly.

    A token crossing the prompt/answer character boundary is rejected. It is
    never silently counted as both prompt and answer. Freeze an unambiguous
    delimiter, tokenizer offsets and truncation before preparing all arms.
    """
    if len(offsets) != len(attention_mask):
        raise ValueError("Tokenizer offsets and attention mask must align")
    result = []
    for (start, end), valid in zip(offsets, attention_mask):
        if valid and start < answer_start < end:
            raise ValueError("Token straddles prompt/answer boundary")
        result.append(bool(valid and start >= answer_start and end > start))
    if not any(result[1:]):
        raise ValueError("Zero answer targets after frozen truncation")
    return result


def eval_quality(model, examples, *, device, window, stride):
    """Full gold targets from immutable dev IDs; article/problem aggregates."""
    results, seen = [], set()
    model.eval()
    for row in examples:
        if row.get("source_split") == "test" or row["unit_id"] in seen:
            raise ValueError("Pilot quality accepts unique development units only")
        seen.add(row["unit_id"])
        ids, targets = row["input_ids"], row["target_mask"]
        if len(ids) != len(targets) or row["domain"] not in {"text", "math"}:
            raise ValueError("Frozen gold-NLL records required")
        total, count = 0., 0
        for w in scoring_windows(len(ids), window=window, stride=stride):
            mask = [bool(targets[p] and w.target_start <= p < w.target_end) for p in range(w.start, w.end)]
            if not any(mask[1:]):
                continue
            token_ids = torch.tensor([ids[w.start:w.end]], dtype=torch.long, device=device)
            target_mask = torch.tensor([mask], dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = model(token_ids, attention_mask=torch.ones_like(target_mask), use_cache=False).logits
                loss = causal_task_loss(logits, token_ids, target_mask)
            total += float(loss.ce_sum)
            count += loss.token_count
        if count != sum(targets[1:]) or count == 0:
            raise ValueError("Gold targets omitted/repeated or empty; retain original denominator")
        results.append({"unit_id":row["unit_id"], "domain":row["domain"],
                        "unit_type":"article" if row["domain"] == "text" else "problem",
                        "nll_sum":total,"target_count":count})
    return {"units":results,"window":window,"stride":stride,"gold_targets_scored_once":True,
            "perplexity":{g: float(torch.exp(torch.tensor(sum(r['nll_sum'] for r in results if r['domain']==g)/
                                                    sum(r['target_count'] for r in results if r['domain']==g))))
                          for g in {r['domain'] for r in results}},
            "metric_scope":"pilot_512" if window == 512 else "confirmation_2048"}
