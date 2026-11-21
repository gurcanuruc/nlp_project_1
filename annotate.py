import json, os, re, sys, textwrap

os.system("")  # Windows'ta ANSI renklerini etkinleştirir
sys.stdout.reconfigure(encoding="utf-8")

PATH = "gold_entities.json"
STOP = {"the", "this", "that", "it", "i", "a", "an", "he", "she", "they", "we", "you",
        "if", "but", "and", "so", "in", "on", "at", "for", "of", "to", "my", "his", "her",
        "there", "what", "when", "who", "how", "why", "as", "or", "is", "was", "not",
        "no", "yes", "one", "all", "some", "just", "even", "also", "after", "before"}

CONNECT = r"(?:of|the|with|de|la|von|van)"
CAP = r"[A-Z][\w'’\-]*(?:\.[A-Z0-9][\w'’\-]*)*"
PATTERN = re.compile(rf"{CAP}(?:\s+(?:{CONNECT}\s+)*{CAP})*")
QUOTE = re.compile(r'"([^"\n]{3,60})"|“([^”\n]{3,60})”|(?<!\w)\'([^\'\n]{3,60})\'(?!\w)')

YELLOW = "\033[30;43m"   # sarı zemin: güçlü öneri
GRAY = "\033[4;90m"      # gri altı çizili: cümle başı tek kelime (zayıf öneri)
CYAN = "\033[1;36m"      # numara etiketi
RESET = "\033[0m"


def norm(text):
    return re.sub(r"\s+", " ", text.replace("<br />", " ")).strip()


def candidates(text):
    strong, weak = [], []

    def clean(c):
        return c.strip(" .,:;!?'\"“”‘’-")

    for m in QUOTE.finditer(text):  # tırnak içindekiler (çoğu zaman film adı)
        c = next(g for g in m.groups() if g).strip()
        if c and c not in strong:
            strong.append(c)

    for m in PATTERN.finditer(text):
        c = clean(m.group())
        if len(c) < 2 or c.lower() in STOP or c in strong or c in weak:
            continue
        before = text[:m.start()].rstrip()
        sentence_start = before == "" or before[-1] in ".!?"
        single = " " not in c
        (weak if (single and sentence_start) else strong).append(c)
    return strong, weak


ANSI = re.compile(r"\033\[[0-9;]*m")


def build(text):
    """Önerileri metindeki görünüş sırasına göre numaralandırır."""
    strong, weak = candidates(text)
    kind = {c: "strong" for c in strong}
    kind.update({c: "weak" for c in weak})
    if not kind:
        return [], kind, []
    alt = "|".join(re.escape(c) for c in sorted(kind, key=len, reverse=True))
    matches = list(re.finditer(rf"(?<!\w)(?:{alt})(?!\w)", text))
    order = []
    for m in matches:
        if m.group(0) not in order:
            order.append(m.group(0))
    return order, kind, matches


def render(text, order, kind, matches):
    out, last, seen = [], 0, set()
    for m in matches:
        c = m.group(0)
        out.append(text[last:m.start()])
        color = YELLOW if kind[c] == "strong" else GRAY
        tag = ""
        if c not in seen:
            seen.add(c)
            tag = f"{CYAN}[{order.index(c) + 1}]{RESET}"
        out.append(f"{color}{c}{RESET}{tag}")
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def wrap(s, width=100):
    """ANSI kodlarını saymadan satır kaydırır."""
    lines, cur, n = [], [], 0
    for tok in s.split(" "):
        w = len(ANSI.sub("", tok))
        if cur and n + 1 + w > width:
            lines.append(" ".join(cur))
            cur, n = [tok], w
        else:
            n = w if not cur else n + 1 + w
            cur.append(tok)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def save(data):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


data = json.load(open(PATH, encoding="utf-8"))
hist = []
i = 0
try:
    while i < len(data):
        ex = data[i]
        if ex["annotated"]:
            i += 1
            continue

        text = norm(ex["text"])
        cands, kind, matches = build(text)
        done = sum(d["annotated"] for d in data)

        print("\n" + "=" * 70)
        print(f"[{done}/{len(data)} tamam]  örnek #{ex['idx']}")
        print(wrap(render(text, cands, kind, matches)))
        print(f"\n{YELLOW} sarı {RESET} = olası isim   {GRAY}gri{RESET} = cümle başı kelime (düşük ihtimal)   "
              f"{CYAN}[n]{RESET} = numara")
        print(textwrap.fill("   ".join(f"{n}) {c}" for n, c in enumerate(cands, 1)), 100))
        print("\nGiriş: numaralar ve/veya yazılan isimler ';' ile ayrılır.  "
              "'-' = entity yok,  s = atla,  b = geri,  q = çık")
        line = input("> ").strip()

        if not line:
            continue
        if line.lower() == "q":
            break
        if line.lower() == "s":
            i += 1
            continue
        if line.lower() == "b":
            prev = [k for k in range(i) if data[k]["annotated"]]
            if prev:
                i = prev[-1]
                print(f"  önceki giriş (#{data[i]['idx']}):", data[i]["entities"] or "(entity yok)")
                data[i]["annotated"] = False
                data[i]["entities"] = []
                save(data)
            else:
                print("  ! geri gidilecek örnek yok")
            continue

        ents = []
        if line != "-":
            for item in line.split(";"):
                item = item.strip()
                if not item:
                    continue
                if item.isdigit() and 1 <= int(item) <= len(cands):
                    item = cands[int(item) - 1]
                elif item.isdigit():
                    print(f"  ! {item} numaralı öneri yok, atlandı")
                    continue
                if item.lower() not in text.lower():
                    print(f"  ! '{item}' metinde bulunamadı, yazım hatası olabilir (yine de eklendi)")
                if item not in ents:
                    ents.append(item)

        ex["entities"] = ents
        ex["annotated"] = True
        hist.append(i)
        save(data)
        print("  kaydedildi:", ents if ents else "(entity yok)")
        i += 1
except KeyboardInterrupt:
    pass

save(data)
print(f"\nToplam etiketlenen: {sum(d['annotated'] for d in data)}/{len(data)}")