import json
import re
import random
import polars as pl

seen = set()
with open("data/processed/finetuning_sample_v3.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line.strip())
        seen.add(row["text"].strip())

df = pl.read_parquet("data/raw/political_discourse_2018_2022.parquet")
print(f"Corpus: {df.height} posts")
df = df.with_columns(
    pl.col("text")
    .str.replace_all(r"https?://\S+|www\.\S+", "")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
).filter(pl.col("text").str.len_chars() > 30)

texts = [t for t in df["text"].to_list() if t.strip() not in seen]
print(f"After removing seen: {len(texts)} posts")

# New terms for acusação antidemocrática
acus_rx = re.compile(
    r"(?i)\b(lawfare|arma[çc][aã]o|persegui[çc][aã]o pol[ií]tica|preso injustamente|"
    r"farsa|juiz parcial|censurou|amorda[çc]ar|"
    r"antidemocr[aá]tic[oa]|amea[çc]a.{0,10}democracia|"
    r"golpista|tentativa de golpe|tramando|tramou|"
    r"fraude.{0,10}eleitoral|fraude.{0,10}urna|manipula[çc][aã]o.{0,10}voto|"
    r"perseguido politicamente|moro parcial|"
    r"condenado injustamente|inocente comprovado|absolvido pela onu)\b"
)

cands_acus2 = [t for t in texts if acus_rx.search(t)]
random.seed(99)
random.shuffle(cands_acus2)
cands_acus2 = cands_acus2[:300]
print(f"acusacao candidates (new terms): {len(cands_acus2)}")

with open("data/processed/cands_acusacao2.jsonl", "w", encoding="utf-8") as f:
    for t in cands_acus2:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

# New terms for ataques políticos
ataq_rx = re.compile(
    r"(?i)\b(vagabundo|safado|covarde|in[uú]til|imprest[aá]vel|"
    r"demagogo|demagogia|traidor|hip[oó]crita|"
    r"mau car[aá]ter|vendido|fracassou|destruiu|arruinou|afundou|"
    r"vergonha nacional|desonest[oa]|imoral|"
    r"irrespons[aá]vel|desgoverno|destrui[çc][aã]o do brasil|"
    r"n[aã]o entrega|omiss[aã]o|omisso|neglig[eê]ncia|neglig[eê]nte|"
    r"mal administrado|p[eé]ssimo presidente|pior governo|governo do caos)\b"
)

cands_ataq2 = [t for t in texts if ataq_rx.search(t)]
random.shuffle(cands_ataq2)
cands_ataq2 = cands_ataq2[:300]
print(f"ataques candidates (new terms): {len(cands_ataq2)}")

with open("data/processed/cands_ataques2.jsonl", "w", encoding="utf-8") as f:
    for t in cands_ataq2:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
