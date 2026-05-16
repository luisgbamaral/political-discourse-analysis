import re

import polars as pl
import torch
from tqdm import tqdm
from transformers import pipeline

from src.utils import BASE_DIR, DATA_RAW, DATA_PROCESSED, get_logger

logger = get_logger("HybridClassifier")

MODEL_DIR = BASE_DIR / "models" / "bert-large-pt-political"

RX_SYSTEMIC_THREATS = re.compile(
    r"(?i)\b(golpe|intervenção militar|artigo 142|invadir|tomem o poder|guerra civil|intervenção federal|estado de sítio|estado de exceção|voto impresso)\b"
)
RX_INSTITUTIONS = re.compile(
    r"(?i)\b(stf|supremo tribunal federal|tse|tribunal eleitoral|alexandre|moraes|xandão|barroso|urna[s]?|voto eletrônico|código.?fonte|sistema eleitoral)\b"
)
# Phrases that signal direct authorial delegitimization of democratic institutions
RX_DIRECT_RHETORIC = re.compile(
    r"(?i)(monocrática|imprópria militância política|usurpação de funções|militância política.*ministro|ministro.*militância política)"
)
RX_ATTRIBUTION = re.compile(
    r"(?i)\b(quer|planeja|planejou|tentou|tenta|ameaçou|incentivou|mentiu|ataca|disse|acusou|afirmou|declarou|alegou|prometeu|defende)\b"
)
RX_POLITICAL_ACTORS = re.compile(
    r"(?i)\b(bolsonaro|lula|presidente|governo|ministro|político|deputado|senador|candidato|\bpt\b|\bpl\b|esquerda|direita)\b"
)
RX_COMMON_INSULTS = re.compile(
    r"(?i)\b(ladrão|ladrões|corrupto|corrupta|incompetente|genocida|rato|lixo|criminoso|presidiário|pior presidente|vergonha|comunista|fascista|bandido|vagabundo)\b"
)
RX_QUOTE_VERBS = re.compile(
    r"(?i)\b(disse|afirmou|declarou|postou|twittou|escreveu|revelou|admitiu|segundo|conforme|de acordo com)\b"
)
RX_NEGATION = re.compile(
    r"(?i)\b(não|nunca|jamais|impediu|bloqueou|evitou|condenou|derrotou|rejeitou|vetou|contra o)\b"
)
# Strong governance signals: vocabulary specific to policy/administration announcements
RX_GOVERNANCE = re.compile(
    r"(?i)\b(disponibilizará|disponibilizou|inaugurou|homologou|sancionou|decretou|regulamentou|"
    r"forças.tarefa|custear|equipamentos adequados|licitação|"
    r"beneficiári[oa]|programa nacional|programa federal|plano nacional|"
    r"obras de|entrega de|investimento[s]? de|recursos para)\b"
)

ZERO_SHOT_LABELS = [
    "retórica antidemocrática",
    "acusação antidemocrática",
    "ataques políticos",
    "administração pública",
    "política econômica",
    "campanha eleitoral",
    "neutro",
]


class DiscourseClassifier:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1

        if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
            logger.info(f"Fine-tuned model found → {MODEL_DIR}")
            self.mode = "finetuned"
            self.pipe = pipeline(
                "text-classification",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
                device=self.device,
                truncation=True,
                max_length=256,
            )
        else:
            fallback = "joeddav/xlm-roberta-large-xnli"
            logger.info(f"No fine-tuned model found. Falling back to zero-shot: {fallback}")
            self.mode = "zero-shot"
            self.pipe = pipeline(
                "zero-shot-classification",
                model=fallback,
                device=self.device,
            )

    def _infer_batch(self, batch: list[str]) -> list[dict]:
        if self.mode == "finetuned":
            preds = self.pipe(batch)
            return [{"label": p["label"], "score": p["score"]} for p in preds]
        preds = self.pipe(
            batch,
            candidate_labels=ZERO_SHOT_LABELS,
            multi_label=False,
            truncation=True,
            hypothesis_template="Este texto é sobre {}.",
        )
        return [{"label": p["labels"][0], "score": p["scores"][0]} for p in preds]

    def _apply_hybrid_logic(self, text: str, model_label: str, model_score: float) -> dict:
        has_systemic_threat  = bool(RX_SYSTEMIC_THREATS.search(text))
        has_institution      = bool(RX_INSTITUTIONS.search(text))
        has_attribution      = bool(RX_ATTRIBUTION.search(text))
        has_actor            = bool(RX_POLITICAL_ACTORS.search(text))
        has_insult           = bool(RX_COMMON_INSULTS.search(text))
        has_quote            = bool(RX_QUOTE_VERBS.search(text))
        has_negation         = bool(RX_NEGATION.search(text))
        has_direct_rhetoric  = bool(RX_DIRECT_RHETORIC.search(text))
        has_governance       = bool(RX_GOVERNANCE.search(text))

        # 1. Direct authorial institutional delegitimization (very specific lexical pattern)
        if has_direct_rhetoric and not has_quote:
            return {"label": "retórica antidemocrática", "score": 0.95}

        # 2. Governance vocabulary overrides insult/attribution signals — policy posts routinely
        #    mention "organizações criminosas", "grupo criminoso", etc. as subjects, not targets
        if has_governance and not has_systemic_threat:
            return {"label": "administração pública", "score": 0.88}

        # 3. Systemic-threat / institution signals
        if has_systemic_threat or has_institution:
            if has_quote or has_negation:
                if has_actor:
                    return {"label": "acusação antidemocrática", "score": 0.85}
                return {"label": model_label, "score": model_score}
            if has_actor and has_attribution:
                return {"label": "acusação antidemocrática", "score": 1.0}
            # Only fire retórica when there is no governance context
            if has_systemic_threat and not has_governance:
                return {"label": "retórica antidemocrática", "score": 1.0}
            if has_institution and has_insult and not has_governance:
                return {"label": "retórica antidemocrática", "score": 1.0}

        # 4. Bare insult directed at a political target
        if has_insult:
            return {"label": "ataques políticos", "score": 1.0}

        return {"label": model_label, "score": model_score}

    def process(self, input_filename: str, output_filename: str) -> None:
        input_path  = DATA_RAW / input_filename
        output_path = DATA_PROCESSED / output_filename

        if not input_path.exists():
            logger.error(f"Input not found: {input_path}")
            return

        logger.info(f"Mode: {self.mode}. Loading and cleaning raw data...")
        df = (
            pl.scan_parquet(input_path)
            .with_columns(
                pl.col("text")
                .str.replace_all(r"https?://\S+|www\.\S+", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
            .filter(pl.col("text").str.len_chars() > 3)
            .collect()
        )

        texts = df["text"].to_list()
        refined_results: list[dict] = []

        logger.info(f"Processing {len(texts)} documents...")

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Hybrid Inference"):
            batch = texts[i: i + self.batch_size]
            try:
                raw_preds = self._infer_batch(batch)
                for doc_text, pred in zip(batch, raw_preds):
                    refined = self._apply_hybrid_logic(doc_text, pred["label"], pred["score"])
                    refined_results.append(refined)
            except Exception as e:
                logger.warning(f"Batch failed at index {i}: {e}")
                refined_results.extend([{"label": "neutro", "score": 0.0}] * len(batch))

        final_df = df.hstack(pl.DataFrame(refined_results))
        final_df.write_parquet(output_path)
        logger.info(f"Success. Data saved to: {output_path}")


if __name__ == "__main__":
    classifier = DiscourseClassifier(batch_size=16)
    classifier.process("political_discourse_2018_2022.parquet", "labeled_discourse_pt.parquet")
