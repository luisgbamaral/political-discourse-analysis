import json
import os
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"

seen = set()
with open("data/processed/finetuning_sample_v3.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line.strip())
        seen.add(row["text"].strip())

df = pl.read_parquet("data/processed/labeled_discourse_pt.parquet")
df = df.with_columns(
    pl.col("text")
    .str.replace_all(r"https?://\S+|www\.\S+", "")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
).filter(
    ~pl.col("text").is_in(list(seen))
).filter(pl.col("text").str.len_chars() > 30)

acus = df.filter(pl.col("label") == "acusação antidemocrática").sort("score")
ataq = df.filter(pl.col("label") == "ataques políticos").sort("score")
border = df.filter(pl.col("score") < 0.5).filter(
    ~pl.col("label").is_in(["acusação antidemocrática", "ataques políticos", "retórica antidemocrática"])
)

counts = {"unseen_total": df.height, "acusacao": acus.height, "ataques": ataq.height, "borderline": border.height}
with open("data/processed/label_dist_unseen.json", "w", encoding="utf-8") as f:
    json.dump(counts, f, ensure_ascii=False, indent=2)

acus.write_ndjson("data/processed/cands_acusacao3.jsonl")
ataq.write_ndjson("data/processed/cands_ataques3.jsonl")
border.write_ndjson("data/processed/cands_borderline.jsonl")
