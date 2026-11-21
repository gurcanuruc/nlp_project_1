import json
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(100))
data = [{"idx": i, "text": ex["text"][:1000], "entities": [], "annotated": False}
        for i, ex in enumerate(ds)]
with open("gold_entities.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)