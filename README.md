# NLP Project I: Structured Text Classification and Information Extraction (local LLM)

Course project (Prof. Umut Orhan, Cukurova University). IMDB movie reviews are turned into JSON
(`category`, `sentiment`, `sentiment_score`, `entities`) with a local open-source LLM served by Ollama.
Two settings are compared: **prompt-only** and **schema-constrained** (`format` = JSON schema).

- Model: `qwen2.5:3b` (Q4_K_M), via Ollama. No cloud APIs.
- Data: `stanfordnlp/imdb` test split, `shuffle(seed=42)`, first 100 reviews, text cut to 1000 characters.
- Report: `NLP_Project1_Report.pdf` (method, annotation rules, matching rule, results, discussion, reproducibility info).

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs one setting at one temperature, writes `results_<setting>_T<temp>.jsonl`. Contains the prompt, JSON schema, `clean()` and validation. |
| `export_for_annotation.py` | Creates `gold_entities.json` (empty annotation template) from the same 100 reviews. |
| `annotate.py` | Interactive helper used to annotate gold entities by hand (highlights candidate names; the final choice is manual). |
| `gold_entities.json` | Manually annotated gold entities for all 100 reviews. |
| `evaluate.py` | Computes raw parse, parse after cleaning, schema validity, accuracy, F1 and entity precision/recall/F1 (exact and partial match). |
| `Analyze_errors.py` | Entity error analysis (false positives and false negatives). |
| `demo.py` | Live test used in the video: new reviews, both settings, temperatures 0.0 and 1.0. |
| `exp_c2_repeat.py` | Defense experiment: same input and seed repeated 5 times, how many distinct outputs. |
| `exp_g1_baseline.py` | Defense experiment: TF-IDF + logistic regression baseline for sentiment. |
| `results_*.jsonl` | Model outputs: prompt-only and schema, each at T=0.0 and T=1.0 (4 files). |

## Reproduce

```bash
pip install ollama datasets scikit-learn
ollama pull qwen2.5:3b            # Ollama must be running

python main.py prompt_only 0.0
python main.py schema 0.0
python main.py prompt_only 1.0
python main.py schema 1.0

python evaluate.py results_prompt_only_T0.0.jsonl results_schema_T0.0.jsonl results_prompt_only_T1.0.jsonl results_schema_T1.0.jsonl
python Analyze_errors.py results_schema_T0.0.jsonl
python demo.py
```

`export_for_annotation.py` and `annotate.py` are only needed to redo the annotation; the finished annotation is in `gold_entities.json`.

## Reproducibility settings

temperature 0.0 and 1.0, seed 42, num_ctx 4096, num_predict 512, other options at Ollama defaults.
Exact versions (Ollama, Python, package) are listed in the report, section 7. Outputs with a fixed seed were identical
in repeated runs in most cases, but exact reproduction is not guaranteed (see the report and `exp_c2_repeat.py`).
