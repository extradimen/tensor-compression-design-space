"""CPU-only deterministic article/problem splits; sealed scanner is separate.

Never fetch data. Callers pass licensed rows; outputs contain source IDs and
content hashes. Raw test contents/answers are not returned by the scanner.
"""
import hashlib
import re
import unicodedata


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def _order(identity, namespace):
    return hashlib.sha256((namespace + "\0" + identity).encode()).hexdigest()


def _bucket(identity, namespace, fractions):
    score = int(_order(identity, namespace), 16) / 2**256
    cumulative = 0.0
    for label, fraction in fractions:
        cumulative += fraction
        if score < cumulative:
            return label
    return fractions[-1][0]


def whole_articles(lines, *, source_split, scanner_only=False):
    if source_split not in ("train", "validation") and not (source_split == "test" and scanner_only is True):
        raise ValueError("Split preparation cannot read the sealed test")
    articles, title, body = [], None, []

    def finish():
        if title is not None:
            text = "\n".join(body)
            digest = content_hash(text)
            articles.append({"unit_id": f"wikitext:{source_split}:{len(articles)}:{digest[:16]}",
                             "title": title, "text": text, "content_sha256": digest,
                             "source_split": source_split})

    for row in lines:
        text = row["text"] if isinstance(row, dict) else row
        for line in text.splitlines():
            heading = re.fullmatch(r"\s*=\s+([^=]+?)\s+=\s*", line)
            if heading:
                finish()
                title, body = heading.group(1).strip(), [line]
            elif title is not None:
                body.append(line)
            elif line.strip():
                raise ValueError("Unassigned text before the first top-level article heading")
    finish()
    return articles


def make_splits(text_train, text_validation, math_train, *, namespace="qai-tensor-v1.1"):
    """Split fixed across seeds. Reserve eight fixed distinct exemplars first.

    Content-equivalent problems are assigned as one cluster, including every
    duplicate of a reserved exemplar. Validation article selection is blind to
    model scores. Rows must contain their original source IDs.
    """
    pools = {"text_calibration": [], "text_repair": [], "text_dev": [],
             "math_calibration": [], "math_repair": [], "math_dev": [], "fewshot": []}
    seen = set()
    for rows in (text_train, text_validation, math_train):
        for row in rows:
            if not row.get("unit_id") or row["unit_id"] in seen or row.get("source_split") == "test":
                raise ValueError("Unique source identities and development-side data required")
            seen.add(row["unit_id"])
    for row in text_train:
        cluster = content_hash(row["text"])
        bucket = _bucket(cluster, namespace + ":text", (("calibration", .25), ("repair", .75)))
        pools["text_" + bucket].append(row["unit_id"])
    distinct_text = {}
    for row in sorted(text_validation, key=lambda r: _order(r["unit_id"], namespace + ":text_dev")):
        distinct_text.setdefault(content_hash(row["text"]), row["unit_id"])
    pools["text_dev"] = list(distinct_text.values())[:64]
    clusters = {}
    for row in math_train:
        # Cluster by question as well as an exact question/answer duplicate:
        # contradictory labels for the same question cannot leak across pools.
        cluster = content_hash(row["question"])
        clusters.setdefault(cluster, []).append(row["unit_id"])
    ordered = sorted(clusters, key=lambda k: _order(k, namespace + ":fewshot"))
    if len(ordered) < 8:
        raise ValueError("Eight distinct fixed exemplars are required")
    reserved = ordered[:8]
    pools["fewshot"] = [sorted(clusters[k])[0] for k in reserved]
    excluded = [unit for k in reserved for unit in clusters[k]]
    for cluster in ordered[8:]:
        bucket = _bucket(cluster, namespace + ":math", (("calibration", .2), ("repair", .6), ("dev", .2)))
        # One representative per independent problem for pilot dev; duplicates
        # in train remain in the same partition and are explicitly recorded.
        units = sorted(clusters[cluster])
        pools["math_" + bucket].extend(units[:1] if bucket == "dev" else units)
    pools["math_dev"] = sorted(pools["math_dev"], key=lambda k: _order(k, namespace + ":math_dev"))[:128]
    for key in pools:
        if key != "fewshot":
            pools[key] = sorted(pools[key])
    return {"namespace": namespace, "pools": pools, "excluded_exemplar_cluster_ids": sorted(excluded),
            "fewshot_content_hashes": [{"unit_id": row["unit_id"], "question_sha256":content_hash(row["question"]),
                                          "answer_sha256":content_hash(row["answer"])}
                                         for row in math_train if row["unit_id"] in pools["fewshot"]],
            "problem_duplicate_clusters": {k: sorted(v) for k, v in sorted(clusters.items()) if len(v) > 1},
            "text_dev_distinct_articles": len(pools["text_dev"]), "text_dev_requested": 64,
            "math_dev_distinct_problems": len(pools["math_dev"]), "math_dev_requested": 128,
            "sealed_test_read": False, "outcome_selection": False}


def blind_contamination_scan(training, development, sealed):
    """Dedicated scanner only; remove training-side offenders, retain eval IDs.

    Freeze NFKC/casefold/whitespace word-token normalization and 13-grams. No
    model outcomes are accepted. Private report stores IDs, never test answers.
    """
    def signatures(text):
        norm = normalized(text)
        tokens = norm.split()
        grams = {content_hash(" ".join(tokens[i:i + 13])) for i in range(len(tokens) - 12)}
        return content_hash(norm), grams

    exact, grams = set(), set()
    for row in sealed:
        e, g = signatures(row["text"])
        exact.add(e)
        grams.update(g)
    flags = {}
    for label, rows in (("training", training), ("development", development)):
        hits = []
        for row in rows:
            e, g = signatures(row["text"])
            if e in exact or g & grams:
                hits.append({"unit_id": row["unit_id"], "exact": e in exact,
                             "overlap_13grams": len(g & grams)})
        flags[label] = hits
    return {"remove_training_ids": sorted(r["unit_id"] for r in flags["training"]),
            "development_flags_retained": flags["development"],
            "sealed_count": len(sealed), "evaluation_denominator_changed": False,
            "normalization": "NFKC_casefold_whitespace_word_tokens", "ngram": 13,
            "test_answers_returned": False}
