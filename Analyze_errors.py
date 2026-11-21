"""Entity hata analizi.
Kullanım:  python analyze_errors.py results_schema_T0.0.jsonl
Çıktı: en sık yanlış (FP) ve kaçırılan (FN) entity'ler + FP'lerin kaba sınıflandırması.
"""
import json, re, sys
from collections import Counter

NAT = {"american", "british", "english", "french", "german", "italian", "russian", "asian",
       "australian", "aussie", "irish", "swiss", "greek", "japanese", "indian", "european"}
GENERIC = {"dvd", "vhs", "tv", "film", "movie", "imdb"}


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(path):
    gold = {g["idx"]: [norm(e) for e in g["entities"]]
            for g in json.load(open("gold_entities.json", encoding="utf-8")) if g["annotated"]}
    fp_all, fn_all = Counter(), Counter()
    cat = Counter()
    tp = fp = fn = 0
    for r in load(path):
        obj = r.get("obj")
        if r["idx"] not in gold:
            continue
        g = set(gold[r["idx"]])
        pred = {norm(e) for e in (obj.get("entities", []) if obj else []) if isinstance(e, str)}
        tp += len(g & pred)
        for e in pred - g:
            fp += 1
            fp_all[e] += 1
            if e in NAT:
                cat["milliyet/grup"] += 1
            elif e in GENERIC:
                cat["genel kelime"] += 1
            elif e == "tom hanks":
                cat["prompt sızıntısı (Tom Hanks)"] += 1
            elif any(e in x or x in e for x in g):
                cat["kısmi eşleşme (sınır farkı)"] += 1
            else:
                cat["diğer (karakter/yer/kurum vb.)"] += 1
        for e in g - pred:
            fn += 1
            fn_all[e] += 1
    print(f"{path}\nTP={tp} FP={fp} FN={fn}\n")
    print("FP sınıfları:")
    for k, v in cat.most_common():
        print(f"  {v:4d}  {k}")
    print("\nEn sık FP:", fp_all.most_common(15))
    print("\nKaçırılan (FN) örnekler:", list(fn_all)[:15])


if __name__ == "__main__":
    main(sys.argv[1])