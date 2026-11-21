import json, re, sys


def norm(s):
    # IMDB İngilizce olduğu için düz lower() yeterli (Türkçe için İ->i, I->ı gerekir)
    return re.sub(r"\s+", " ", str(s).strip().lower())


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def match(p, g):
    return p == g or (p and g and (p in g or g in p))


def entity_metrics(preds, golds):
    tp = fp = fn = 0
    p_hit = g_hit = p_tot = g_tot = 0
    for P, G in zip(preds, golds):
        P, G = {norm(x) for x in P}, {norm(x) for x in G}
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
        p_tot += len(P); g_tot += len(G)
        p_hit += sum(any(match(p, g) for g in G) for p in P)
        g_hit += sum(any(match(p, g) for p in P) for g in G)
    pe, re_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    pp, rp = p_hit / max(p_tot, 1), g_hit / max(g_tot, 1)
    return {"exact": (round(pe, 3), round(re_, 3), round(f1(pe, re_), 3)),
            "partial": (round(pp, 3), round(rp, 3), round(f1(pp, rp), 3))}


def evaluate(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(recs)
    valid = [r for r in recs if r["schema_ok"]]
    pred = {r["idx"]: (1 if r["obj"]["sentiment"] == "positive" else 0) for r in valid}

    correct = sum(pred.get(r["idx"]) == r["label"] for r in recs)
    correct_valid = sum(pred[r["idx"]] == r["label"] for r in valid)
    tp = sum(pred.get(r["idx"]) == 1 and r["label"] == 1 for r in recs)
    fp = sum(pred.get(r["idx"]) == 1 and r["label"] == 0 for r in recs)
    fn = sum(r["label"] == 1 and pred.get(r["idx"]) != 1 for r in recs)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    res = {
        "file": path, "n": n,
        "raw_parse": f"{sum(r['raw_ok'] for r in recs)/n:.1%}",
        "parse_after_cleaning": f"{sum(r['clean_ok'] for r in recs)/n:.1%}",
        "schema_valid": f"{len(valid)/n:.1%}",
        "accuracy_all (invalid = wrong)": f"{correct/n:.1%}",
        "accuracy_on_valid": f"{correct_valid/max(len(valid),1):.1%}",
        "f1_positive_class": round(f1(pr, rc), 3),
        "mean_ms": round(sum(r["ms"] for r in recs) / n),
    }

    try:
        gold = [g for g in json.load(open("gold_entities.json", encoding="utf-8")) if g["annotated"]]
        by_idx = {r["idx"]: r for r in recs}
        preds = [by_idx[g["idx"]]["obj"]["entities"] if by_idx[g["idx"]]["schema_ok"] else []
                 for g in gold]
        res["entity_samples"] = len(gold)
        res["entity (P, R, F1)"] = entity_metrics(preds, [g["entities"] for g in gold]) if gold else "etiket yok"
    except FileNotFoundError:
        res["entity (P, R, F1)"] = "gold_entities.json yok"
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(evaluate(p), indent=2, ensure_ascii=False))