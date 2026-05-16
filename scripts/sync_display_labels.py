"""
After re-running src/inference (which updates labeled_discourse_pt.parquet),
run this script to sync the fresh Portuguese labels into labeled_v4_discourse.parquet.
"""
import polars as pl

display = pl.read_parquet("data/processed/labeled_v4_discourse.parquet")
fresh   = pl.read_parquet("data/processed/labeled_discourse_pt.parquet").select(
    ["text", "label", "score"]
)

updated = (
    display
    .drop(["label", "score"])
    .join(fresh, on="text", how="left")
    .with_columns([
        pl.col("label").fill_null(pl.lit("neutro")),
        pl.col("score").fill_null(pl.lit(0.0)),
    ])
)

updated.write_parquet("data/processed/labeled_v4_discourse.parquet")

counts = updated.group_by("label").len().sort("len", descending=True)
with open("data/processed/label_sync_report.json", "w", encoding="utf-8") as f:
    import json
    json.dump(counts.to_dicts(), f, ensure_ascii=False, indent=2)

print(f"Synced {updated.height} rows")
print(counts)
