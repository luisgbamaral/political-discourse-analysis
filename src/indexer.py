import polars as pl
from src.utils import DATA_PROCESSED, get_logger

logger = get_logger("Indexer")

INPUT_FILE  = "labeled_discourse_pt.parquet"
OUTPUT_FILE = "adi_index.parquet"


def compute_adi(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns([
            pl.col("date").dt.truncate("1mo").alias("month_date"),
            pl.col("date").dt.strftime("%m/%Y").alias("month_str"),
            pl.when(pl.col("label") == "acusação antidemocrática")
              .then(pl.col("score")).otherwise(0.0).alias("accusation_score"),
            pl.when(pl.col("label") == "ataques políticos")
              .then(pl.col("score")).otherwise(0.0).alias("attacks_score"),
        ])
        .group_by(["author", "month_date", "month_str"])
        .agg([
            pl.len().alias("total_posts"),
            pl.col("accusation_score").sum().alias("accusation_score_sum"),
            pl.col("attacks_score").sum().alias("attacks_score_sum"),
        ])
        .with_columns([
            (pl.col("accusation_score_sum") / pl.col("total_posts") * 100).alias("accusation_monthly"),
            (pl.col("attacks_score_sum")    / pl.col("total_posts") * 100).alias("attacks_monthly"),
        ])
        .sort(["author", "month_date"])
        .with_columns([
            pl.col("accusation_monthly").cum_sum().over("author").alias("accusation_cumulative"),
            pl.col("attacks_monthly").cum_sum().over("author").alias("attacks_cumulative"),
        ])
    )


def main() -> None:
    input_path  = DATA_PROCESSED / INPUT_FILE
    output_path = DATA_PROCESSED / OUTPUT_FILE

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run src/inference.py first.")
        return

    df  = pl.read_parquet(input_path)
    adi = compute_adi(df)

    adi.write_parquet(output_path)
    logger.info(f"ADI index saved → {output_path}")
    logger.info(f"\n{adi.sort(['author', 'month_date'])}")


if __name__ == "__main__":
    main()
