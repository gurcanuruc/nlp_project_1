"""Savunma deneyi C2: aynı girdi, aynı seed, art arda N kez -> çıktılar aynı mı?

Kullanım:  python exp_c2_repeat.py
Her örnek için iki ayarda (prompt_only, schema) ve iki sıcaklıkta (0.0, 1.0)
N=5 kez çağırır, kaç FARKLI çıktı geldiğini yazar.
main.py'deki ask() kullanılır (seed=42 sabit).
"""
import sys
from datasets import load_dataset
import main as M

N_REPEAT = 5
N_SAMPLES = 5            # ilk 5 örnek yeterli, hızlı biter
TEMPS = [0.0, 1.0]
SETTINGS = ["prompt_only", "schema"]


def get_raw(text, setting, temp):
    out = M.ask(text, setting, temp)
    return out[0] if isinstance(out, tuple) else out


def main():
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42)
    texts = [ds[i]["text"] for i in range(N_SAMPLES)]
    print("Ilk istek model yukleme nedeniyle yavas olabilir...\n")
    for temp in TEMPS:
        for setting in SETTINGS:
            print(f"--- T={temp} | {setting} | seed={M.SEED} | {N_REPEAT} tekrar ---")
            total_distinct = 0
            for i, t in enumerate(texts):
                outs = [get_raw(t, setting, temp) for _ in range(N_REPEAT)]
                distinct = len(set(outs))
                total_distinct += distinct
                flag = "AYNI" if distinct == 1 else f"{distinct} FARKLI"
                print(f"  ornek {i}: {flag}")
                if distinct > 1:
                    for o in sorted(set(outs)):
                        print("      ", o[:140].replace("\n", " "))
            print(f"  toplam: {total_distinct}/{N_SAMPLES * 1} (her ornek icin 1 = tamamen tekrarlanabilir)\n")


if __name__ == "__main__":
    main()