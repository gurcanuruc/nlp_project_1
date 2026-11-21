"""A1 demosu: clean() icindeki raw_decode satiri olmadan ne bozulur?

Kullanim:  python demo_a1_clean.py
results_prompt_only_T0.0.jsonl icindeki #50 numarali ornegi alir
(model ciktisi fazladan bir '}' ile bitiyor) ve uc yolu karsilastirir:
  1) ham json.loads
  2) main.clean() (raw_decode ile)
  3) raw_decode OLMADAN yazilmis clean (sadece rfind('}') yedegi)
"""
import json, re
import main as M

IDX = 50
FILE = "results_prompt_only_T0.0.jsonl"


def clean_without_raw_decode(raw):
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    s = raw.find("{")
    if s == -1:
        return raw
    e = raw.rfind("}")
    return raw[s:e + 1] if e != -1 else raw[s:]


def try_load(label, text):
    try:
        json.loads(text)
        print(f"{label:38} -> BASARILI")
    except Exception as ex:
        print(f"{label:38} -> HATA: {ex}")


rec = next(r for r in map(json.loads, open(FILE, encoding="utf-8")) if r["idx"] == IDX)
raw = rec["raw"]
print("Ham cikti (son kisim):", repr(raw[-70:]), "\n")
try_load("1) json.loads(raw)", raw)
try_load("2) clean() - raw_decode ile", M.clean(raw))
try_load("3) clean - raw_decode OLMADAN", clean_without_raw_decode(raw))