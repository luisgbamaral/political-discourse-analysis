import re

import polars as pl
import torch
from tqdm import tqdm
from transformers import pipeline

from src.utils import BASE_DIR, DATA_RAW, DATA_PROCESSED, get_logger

logger = get_logger("HybridClassifier")

MODEL_DIR = BASE_DIR / "models" / "bert-large-pt-political"

RX_SYSTEMIC_THREATS = re.compile(
    r"(?i)\b("
    r"golpe\s+de\s+estado|tentativa\s+de\s+golpe|golpe\s+institucional|golpismo|golpista|"
    r"intervenção\s+militar|artigo\s+142|tomem\s+o\s+poder|guerra\s+civil|"
    r"intervenção\s+federal|estado\s+de\s+sítio|estado\s+de\s+exceção|voto\s+impresso"
    r")\b"
)
RX_INSTITUTIONS = re.compile(
    r"(?i)\b("
    r"stf|supremo\s+tribunal\s+federal|tse|tribunal\s+eleitoral|"
    r"moraes|xandão|barroso|urna[s]?|voto\s+eletrônico|código.{0,2}fonte|sistema\s+eleitoral"
    r")\b"
)
# Direct authorial delegitimization — very specific institutional attack phrases
RX_DIRECT_RHETORIC = re.compile(
    r"(?i)("
    r"monocrática|imprópria\s+militância\s+política|usurpação\s+de\s+funções|"
    r"militância\s+política.{0,20}ministro|ministro.{0,20}militância\s+política"
    r")"
)
# Verbs that attribute a specific antidemocratic action to an actor
RX_ATTRIBUTION = re.compile(
    r"(?i)\b(tentou|tenta|ameaçou|ameaça|incentivou|mentiu|ataca|atacou|acusou|alegou|tramou|tramando)\b"
)
RX_POLITICAL_ACTORS = re.compile(
    r"(?i)\b(bolsonaro|lula|presidente|ministro|deputado|senador|candidato|\bpt\b|\bpl\b)\b"
)
# Personal insults directed at political figures — restricted to terms rarely used in neutral contexts
RX_COMMON_INSULTS = re.compile(
    r"(?i)\b(ladrão|ladrões|corrupto|corrupta|incompetente|genocida|rato|presidiário|pior\s+presidente|vagabundo)\b"
)
RX_QUOTE_VERBS = re.compile(
    r"(?i)\b(disse|afirmou|declarou|postou|twittou|escreveu|revelou|admitiu|conforme|de\s+acordo\s+com)\b"
)
# Specific negation/opposition verbs — excludes bare "não" which appears in 90%+ of posts
RX_NEGATION = re.compile(
    r"(?i)\b(nunca|jamais|impediu|bloqueou|evitou|condenou|derrotou|rejeitou|vetou|contra\s+o)\b"
)
# Governance vocabulary: policy announcements, law enforcement operations, public programs
RX_GOVERNANCE = re.compile(
    r"(?i)\b("
    r"disponibilizará|disponibilizou|inaugurou|homologou|sancionou|decretou|regulamentou|"
    r"forças.{0,3}tarefa|custear|licitação|"
    r"beneficiári[oa]|programa\s+(nacional|federal|estadual)|plano\s+nacional|"
    r"obras\s+de|entrega\s+de|investimento[s]?\s+de|recursos\s+para|"
    r"polícia\s+federal|mandado[s]?\s+de\s+(busca|prisão)|busca\s+e\s+apreensão|"
    r"ministério\s+d[ao]|secretaria\s+(nacional|de)|"
    r"coleta\s+seletiva|logística\s+reversa|reciclagem|saneamento\s+básico|"
    r"marco\s+(legal|regulatório)|operação\s+\w+"
    r")\b"
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
            if has_actor and has_attribution and not has_governance:
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
