import json, re, sys
import ollama
from datasets import load_dataset

MODEL = "qwen2.5:3b"
N = 100
SEED = 42
NUM_CTX = 4096
NUM_PREDICT = 512

SYSTEM = """You are an information extraction system for movie reviews.
Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
Schema:
{"category": "movie_review",
 "sentiment": "positive" or "negative",
 "sentiment_score": float between 0 and 1 (1 = very positive),
 "entities": [names of movies, actors, directors mentioned]}

Example output:
{"category": "movie_review", "sentiment": "negative", "sentiment_score": 0.1, "entities": ["Tom Hanks"]}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["movie_review"]},
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "sentiment_score": {"type": "number"},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "sentiment", "sentiment_score", "entities"],
}


def ask(text, setting, temperature):
    # prompt_only: format parametresi YOK; schema: format=JSON schema
    kwargs = {"format": SCHEMA} if setting == "schema" else {}
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text[:1000]}],
        options={"temperature": temperature, "seed": SEED,
                 "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        **kwargs,
    )
    ms = (getattr(r, "total_duration", 0) or 0) / 1e6
    return r["message"]["content"], ms


def clean(raw):
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    s = raw.find("{")
    if s == -1:
        return raw
    try:
        # ilk tam JSON nesnesini al, sondaki fazlalığı at
        _, end = json.JSONDecoder().raw_decode(raw[s:])
        return raw[s:s + end]
    except json.JSONDecodeError:
        e = raw.rfind("}")
        return raw[s:e + 1] if e != -1 else raw[s:]


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(obj):
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("category"), str)
        and obj.get("sentiment") in ("positive", "negative")
        and is_num(obj.get("sentiment_score"))
        and 0 <= obj["sentiment_score"] <= 1
        and isinstance(obj.get("entities"), list)
        and all(isinstance(e, str) for e in obj["entities"])
    )


def run(setting, temperature):
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(N))
    out = f"results_{setting}_T{temperature}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            try:
                raw, ms = ask(ex["text"], setting, temperature)
            except Exception as e:  # Ollama bağlantı/model hatası
                raw, ms = f"__REQUEST_ERROR__: {e}", 0

            rec = {"idx": i, "label": ex["label"], "raw": raw, "ms": round(ms),
                   "raw_ok": False, "clean_ok": False, "schema_ok": False}

            try:                                   # 1) temizlemeden, ham çıktı
                json.loads(raw)
                rec["raw_ok"] = True
            except json.JSONDecodeError:
                pass

            try:                                   # 2) temizleme sonrası
                obj = json.loads(clean(raw))
                rec["clean_ok"] = True
                if validate(obj):
                    rec["schema_ok"] = True
                    rec["obj"] = obj
            except json.JSONDecodeError:
                pass

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{setting} T={temperature}: {i+1}/{N}", end="\r")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    # python main.py prompt_only 0   |   python main.py schema 1.0
    run(sys.argv[1], float(sys.argv[2]))