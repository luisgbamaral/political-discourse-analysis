import os
import torch
import polars as pl
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils import get_logger, DATA_PROCESSED

load_dotenv()
logger = get_logger("DisplayTranslator")


class DisplayTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-roa-en", batch_size: int = 4):
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = model_name

        logger.info(f"Initializing translator: {model_name} (Device: {self.device})")

        self.pipeline = pipeline(
            "translation",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            clean_up_tokenization_spaces=True,
            token=os.getenv("HF_TOKEN"),
        )

    def _chunk_text(self, text: str, max_chars: int = 1000) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in text.split(". "):
            clean_sent = sentence.strip() + "."
            sent_len = len(clean_sent)

            if current_len + sent_len > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [clean_sent]
                current_len = sent_len
            else:
                current_chunk.append(clean_sent)
                current_len += sent_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process(self, input_filename: str, output_filename: str) -> None:
        input_path     = DATA_PROCESSED / input_filename
        output_path    = DATA_PROCESSED / output_filename
        checkpoint_path = DATA_PROCESSED / (output_filename.replace(".parquet", "_chunks_checkpoint.parquet"))

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        df = pl.read_parquet(input_path)
        texts = df["text"].to_list()

        mapped_inputs: list[tuple[int, str]] = []
        for idx, text in enumerate(texts):
            for chunk in self._chunk_text(text):
                mapped_inputs.append((idx, chunk))

        chunks_to_translate = [m[1] for m in mapped_inputs]
        checkpoint_every = 100

        if checkpoint_path.exists():
            df_chunks = pl.read_parquet(checkpoint_path)
            already_done = df_chunks.height
            logger.info(f"Checkpoint encontrado: {already_done}/{len(chunks_to_translate)} chunks traduzidos. Retomando...")
            translated_chunks: list[str] = df_chunks["chunk_trans"].to_list()
            start_idx = (already_done // self.batch_size) * self.batch_size
        else:
            translated_chunks = []
            start_idx = 0

        if start_idx < len(chunks_to_translate):
            logger.info(f"Traduzindo {len(chunks_to_translate) - start_idx} chunks restantes...")
            for i in tqdm(range(start_idx, len(chunks_to_translate), self.batch_size), desc="Translating"):
                batch = chunks_to_translate[i: i + self.batch_size]
                try:
                    results = self.pipeline(batch, truncation=True, max_length=512)
                    translated_chunks.extend([r["translation_text"] for r in results])
                except Exception as e:
                    logger.error(f"Batch failed at index {i}: {e}")
                    translated_chunks.extend([""] * len(batch))

                if (i + self.batch_size) % checkpoint_every == 0 or i + self.batch_size >= len(chunks_to_translate):
                    limit = min(len(mapped_inputs), len(translated_chunks))
                    pl.DataFrame({
                        "original_idx": pl.Series([m[0] for m in mapped_inputs[:limit]], dtype=pl.UInt32),
                        "chunk_trans":  translated_chunks[:limit],
                    }).write_parquet(checkpoint_path)

        limit = min(len(mapped_inputs), len(translated_chunks))
        df_chunks = pl.DataFrame({
            "original_idx": pl.Series([m[0] for m in mapped_inputs[:limit]], dtype=pl.UInt32),
            "chunk_trans":  translated_chunks[:limit],
        })

        df_translated = (
            df_chunks
            .group_by("original_idx", maintain_order=True)
            .agg(pl.col("chunk_trans").str.concat(" "))
            .rename({"chunk_trans": "text_en"})
        )

        final_df = (
            df.with_row_index(name="original_idx")
            .join(df_translated, on="original_idx", how="left")
            .drop("original_idx")
            .fill_null("")
        )

        final_df.write_parquet(output_path)
        logger.info(f"Saved display dataset to {output_path}")
        checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    translator = DisplayTranslator(batch_size=4)
    translator.process("labeled_discourse_pt.parquet", "labeled_v4_discourse.parquet")
