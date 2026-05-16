"""
After re-running src/inference (which updates labeled_discourse_pt.parquet),
run this script to sync the fresh Portuguese labels into labeled_v4_discourse.parquet.
"""
import polars as pl

display = pl.read_parquet("data/processed/labeled_v4_discourse.parquet")

# Deduplicate PT by cleaned text, keeping highest-confidence label
fresh = (
    pl.read_parquet("data/processed/labeled_discourse_pt.parquet")
    .select(["text", "label", "score"])
    .sort("score", descending=True)
    .unique(subset=["text"], keep="first")
    .rename({"text": "text_clean_pt"})
)

updated = (
    display
    .drop(["label", "score"])
    .join(fresh, on="text_clean_pt", how="left")
    .with_columns([
        pl.col("label").fill_null(pl.lit("neutro")),
        pl.col("score").fill_null(pl.lit(0.0)),
    ])
)

assert updated.height == display.height, f"Row count changed: {display.height} -> {updated.height}"

updated.write_parquet("data/processed/labeled_v4_discourse.parquet")

counts = updated.group_by("label").len().sort("len", descending=True)
import json
with open("data/processed/label_sync_report.json", "w", encoding="utf-8") as f:
    json.dump(counts.to_dicts(), f, ensure_ascii=False, indent=2)

print(f"Synced {updated.height} rows")
print(counts)
