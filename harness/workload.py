"""Build a fixed, gradable workload from OPEN data that mirrors the GreenMind trace mix.

Classes and default counts (same as the hidden Final track):
  chat 90 (MMLU), rag 60 (needle-in-docs), agentic 60 (multi-turn fake tools),
  reasoning 45 (GSM8K), thai 30 (Belebele tha_Thai), long 15 (needle at 18-28K tokens)

  python harness/workload.py --out data/workload.jsonl            # 300 items
  python harness/workload.py --out data/val.jsonl --scale 0.34     # ~100 items
  python harness/workload.py --out data/mock.jsonl --synthetic     # no downloads (testing)

Nothing here touches the hidden test set; every item is generated or drawn from public datasets.
"""
import argparse
import hashlib
import json
import random

CLASS_MIX = {"chat": 90, "rag": 60, "agentic": 60, "reasoning": 45, "thai": 30, "long": 15}
LETTERS = "ABCD"

# ---------------------------------------------------------------- fake tools (deterministic)
CITIES = ["Bangkok", "Chiang Mai", "Oslo", "Tokyo", "Nairobi", "Lima", "Yangon", "Berlin",
          "Sydney", "Toronto", "Cairo", "Hanoi", "Madrid", "Seoul", "Phuket", "Denver"]
RATES = {"USD": 1.0, "THB": 36.0, "EUR": 0.92, "JPY": 150.0, "MMK": 2100.0, "GBP": 0.79}
STATUSES = ["shipped", "processing", "delivered", "cancelled"]

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current temperature in Celsius for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount between currencies (ISO codes).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "lookup_order", "description": "Look up an order's status and item count by order id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
]


def _h(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def call_tool(name, args):
    """Deterministic fake tool backend, shared by the builder (expected answers) and the runner."""
    if name == "get_weather":
        city = str(args.get("city", "")).strip().title()
        return {"city": city, "temp_c": _h(city) % 35 + 1}
    if name == "convert_currency":
        f, t = str(args.get("from_currency", "")).upper(), str(args.get("to_currency", "")).upper()
        if f not in RATES or t not in RATES:
            return {"error": "unknown currency"}
        return {"amount": round(float(args.get("amount", 0)) / RATES[f] * RATES[t], 2), "currency": t}
    if name == "lookup_order":
        oid = str(args.get("order_id", "")).strip().upper()
        return {"order_id": oid, "status": STATUSES[_h(oid) % 4], "items": _h(oid + "n") % 9 + 1}
    return {"error": f"unknown tool {name}"}


def agentic_items(n, rng):
    out = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a, b = rng.sample(CITIES, 2)
            ta, tb = call_tool("get_weather", {"city": a})["temp_c"], call_tool("get_weather", {"city": b})["temp_c"]
            q = f"Use the tools. How many degrees Celsius warmer is {a} than {b} right now? Negative if colder. Give the final number."
            ans = ta - tb
        elif kind == 1:
            amt = rng.choice([25, 80, 120, 450, 1000])
            f, t = rng.sample(list(RATES), 2)
            q = f"Use the tools. Convert {amt} {f} to {t}, then tell me the result rounded to 2 decimals."
            ans = call_tool("convert_currency", {"amount": amt, "from_currency": f, "to_currency": t})["amount"]
        else:
            ids = [f"ORD-{rng.randint(1000, 9999)}" for _ in range(3)]
            q = (f"Use the tools. Look up orders {', '.join(ids)}. "
                 "How many items are there in total across those orders? Give the final number.")
            ans = sum(call_tool("lookup_order", {"order_id": o})["items"] for o in ids)
        out.append({"cls": "agentic", "messages": [{"role": "user", "content": q}], "tools": TOOLS,
                    "grade": {"type": "number", "answer": ans}})
    return out


# ---------------------------------------------------------------- needle-in-docs (rag / long)
CODENAMES = ["Amber", "Basalt", "Cobalt", "Delta", "Ember", "Fjord", "Garnet", "Harbor", "Indigo",
             "Juniper", "Kestrel", "Lotus", "Mango", "Nimbus", "Onyx", "Pine", "Quartz", "Raven"]


def needle_items(n, rng, cls, tok_lo, tok_hi, passages):
    out = []
    for _ in range(n):
        target = rng.randint(tok_lo, tok_hi)
        names = rng.sample(CODENAMES, 6)
        facts = {nm: rng.randint(1000, 99999) for nm in names}
        chunks, words = [], 0
        while words * 1.35 < target:  # ~1.35 tokens per English word
            p = rng.choice(passages)
            chunks.append(p)
            words += len(p.split())
        for nm, code in facts.items():
            chunks.insert(rng.randrange(len(chunks) + 1), f"Internal memo: the budget code of project {nm} is {code}.")
        doc = "\n\n".join(f"[Document {i + 1}]\n{c}" for i, c in enumerate(chunks))
        ask = rng.choice(names)
        q = f"{doc}\n\nQuestion: According to the documents above, what is the budget code of project {ask}? Answer with the number."
        out.append({"cls": cls, "messages": [{"role": "user", "content": q}],
                    "grade": {"type": "number", "answer": facts[ask]}})
    return out


# ---------------------------------------------------------------- public datasets
def mmlu_items(n, rng):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        q = r["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, r["choices"]))
        q += "\n\nAnswer with the letter of the correct option."
        out.append({"cls": "chat", "messages": [{"role": "user", "content": q}],
                    "grade": {"type": "choice", "answer": LETTERS[r["answer"]], "options": LETTERS}})
    return out


def gsm8k_items(n, rng):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        ans = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        out.append({"cls": "reasoning", "messages": [{"role": "user", "content": r["question"] + "\nGive the final answer as a number."}],
                    "grade": {"type": "number", "answer": ans}})
    return out


def belebele(lang):
    from datasets import load_dataset
    return load_dataset("facebook/belebele", lang, split="test")


def thai_items(n, rng):
    ds = belebele("tha_Thai")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        opts = "\n".join(f"{k}. {r[f'mc_answer{k}']}" for k in range(1, 5))
        q = (f"{r['flores_passage']}\n\nคำถาม: {r['question']}\n{opts}\n\n"
             "ตอบด้วยหมายเลขของตัวเลือกที่ถูกต้อง (1-4)")
        out.append({"cls": "thai", "messages": [{"role": "user", "content": q}],
                    "grade": {"type": "choice", "answer": str(r["correct_answer_num"]), "options": "1234"}})
    return out


def english_passages():
    return sorted(set(belebele("eng_Latn")["flores_passage"]))


# ---------------------------------------------------------------- synthetic fallback (no network)
def synthetic(counts, rng):
    filler = [" ".join(rng.choice(["solar", "grid", "load", "battery", "carbon", "wind", "power", "meter",
                                   "district", "report", "quarterly", "efficiency"]) for _ in range(120))
              for _ in range(50)]
    out = []
    for i in range(counts["chat"]):
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        opts = [a + b, a + b + 1, a + b - 1, a * b]
        rng.shuffle(opts)
        q = f"What is {a} + {b}?\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, opts)) + "\nAnswer with the letter."
        out.append({"cls": "chat", "messages": [{"role": "user", "content": q}],
                    "grade": {"type": "choice", "answer": LETTERS[opts.index(a + b)], "options": LETTERS}})
    for i in range(counts["reasoning"]):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        out.append({"cls": "reasoning", "messages": [{"role": "user", "content": f"Compute {a}*{b}. Give the number."}],
                    "grade": {"type": "number", "answer": a * b}})
    for i in range(counts["thai"]):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        out.append({"cls": "thai", "messages": [{"role": "user", "content": f"{a} บวก {b} เท่ากับเท่าไร? 1. {a+b} 2. {a+b+2} 3. {a*b+1} 4. 0 ตอบด้วยหมายเลข"}],
                    "grade": {"type": "choice", "answer": "1", "options": "1234"}})
    out += needle_items(counts["rag"], rng, "rag", 2000, 7000, filler)
    out += needle_items(counts["long"], rng, "long", 18000, 28000, filler)
    out += agentic_items(counts["agentic"], rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="0.34 ~ validation-sized (100 items)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--synthetic", action="store_true", help="no dataset downloads; for harness testing only")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    counts = {k: max(1, round(v * a.scale)) for k, v in CLASS_MIX.items()}

    if a.synthetic:
        items = synthetic(counts, rng)
    else:
        passages = english_passages()
        items = (mmlu_items(counts["chat"], rng) + needle_items(counts["rag"], rng, "rag", 2000, 7000, passages)
                 + agentic_items(counts["agentic"], rng) + gsm8k_items(counts["reasoning"], rng)
                 + thai_items(counts["thai"], rng) + needle_items(counts["long"], rng, "long", 18000, 28000, passages))
    rng.shuffle(items)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            it["id"] = i
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {a.out}: {counts}")


if __name__ == "__main__":
    main()
