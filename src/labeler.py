import json
import os
import time
from pathlib import Path

import polars as pl
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import DATA_PROCESSED, get_logger

load_dotenv()
logger = get_logger("Labeler")

INPUT_FILE     = "labeled_discourse_pt.parquet"
OUTPUT_PARQUET = "finetuning_sample.parquet"
OUTPUT_JSONL   = "finetuning_sample.jsonl"

VALID_LABELS = {
    "retórica antidemocrática",
    "acusação antidemocrática",
    "ataques políticos",
    "administração pública",
    "política econômica",
    "campanha eleitoral",
    "neutro",
}

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """Você é um especialista em análise do discurso político brasileiro.
Sua tarefa é classificar mensagens extraídas dos canais oficiais do Telegram de Lula e Bolsonaro
durante a campanha eleitoral de 2022. Retorne APENAS o rótulo correto, sem explicações adicionais.

## Categorias disponíveis

**retórica antidemocrática**
Texto que, de forma direta e autoral, ataca instituições ou convoca ações antidemocráticas.
Exemplos: ameaças ao STF, defesa de intervenção militar, dúvidas sobre urnas eletrônicas
apresentadas como fatos, chamados a invadir prédios públicos.
Pista: O próprio canal está fazendo a afirmação ou convocação, não citando terceiros.

**acusação antidemocrática**
Texto que denuncia ou acusa o adversário de comportamento antidemocrático, mas não constitui
ele mesmo uma ameaça sistêmica. O autor está narrando/acusando, não executando.
Exemplos: "Bolsonaro planeja um golpe", "O PT está destruindo a democracia".

**ataques políticos**
Críticas pessoais, morais ou de competência ao adversário ou a aliados políticos.
Não há ameaça institucional; é retórica agressiva comum na disputa eleitoral.
Exemplos: "Lula é ladrão", "Bolsonaro é genocida", xingamentos e apelidos pejorativos.

**administração pública**
Conteúdo sobre ações de governo, políticas públicas, decisões ministeriais, legislação,
orçamento ou desempenho da gestão pública — sem cunho de ataque pesado.

**política econômica**
Conteúdo focado em economia: inflação, emprego, auxílio social, combustível, privatizações,
reforma tributária, câmbio, PIB.

**campanha eleitoral**
Conteúdo de mobilização eleitoral: agenda de eventos, pedidos de voto, divulgação de pesquisas,
resultados eleitorais, estratégias de campanha, material de propaganda sem ataques graves.

**neutro**
Conteúdo informativo, institucional ou que não se encaixa claramente em nenhuma das categorias
acima: divulgação de agenda, agradecimentos, links sem contexto, mensagens muito curtas.

## Regra de ouro: Retórica vs Acusação
- Se o canal É O AUTOR da ameaça/afirmação problemática → **retórica antidemocrática**
- Se o canal RELATA ou ACUSA o adversário → **acusação antidemocrática**
- Verbos de citação (disse, afirmou, declarou, segundo, conforme) → sinal forte de acusação
- Negação + ameaça (bloqueou, impediu, condenou o golpe) → provável acusação ou neutro

## Formato de resposta
Responda com exatamente um dos rótulos abaixo (sem aspas, sem pontuação extra):
retórica antidemocrática
acusação antidemocrática
ataques políticos
administração pública
política econômica
campanha eleitoral
neutro"""


def build_sample(df: pl.DataFrame) -> pl.DataFrame:
    collected: list[pl.DataFrame] = []

    rhetoric = df.filter(pl.col("label") == "retórica antidemocrática")
    if rhetoric.height > 200:
        rhetoric = rhetoric.sample(200, seed=42)
    collected.append(rhetoric)

    accusation = df.filter(pl.col("label") == "acusação antidemocrática")
    if accusation.height > 200:
        accusation = accusation.sample(200, seed=42)
    collected.append(accusation)

    low_conf = df.filter(pl.col("score") < 0.65)
    if low_conf.height > 200:
        low_conf = low_conf.sample(200, seed=42)
    collected.append(low_conf)

    for label in VALID_LABELS - {"retórica antidemocrática", "acusação antidemocrática"}:
        subset = df.filter(pl.col("label") == label)
        if subset.height > 50:
            subset = subset.sample(50, seed=42)
        collected.append(subset)

    combined = pl.concat(collected).unique(subset=["text"])
    logger.info(f"Sample built: {combined.height} rows")
    return combined


def call_claude(client: Anthropic, text: str) -> tuple[str, str | None]:
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=32,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Classifique esta mensagem:\n\n{text[:2000]}",
                }
            ],
        )
        raw = response.content[0].text.strip().lower()
        for label in VALID_LABELS:
            if label in raw:
                return raw, label
        logger.warning(f"Unrecognised label: '{raw}' — falling back to neutro")
        return raw, "neutro"
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return "", None


def label_sample(sample: pl.DataFrame) -> pl.DataFrame:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    gold_labels: list[str] = []
    raw_responses: list[str] = []

    for i, text in enumerate(tqdm(sample["text"].to_list(), desc="Labeling with Claude")):
        raw, label = call_claude(client, text)
        gold_labels.append(label if label else "neutro")
        raw_responses.append(raw)
        if (i + 1) % 50 == 0:
            time.sleep(2)

    return sample.with_columns([
        pl.Series("gold_label", gold_labels),
        pl.Series("claude_raw", raw_responses),
    ])


def save_jsonl(df: pl.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            f.write(json.dumps({"text": row["text"], "label": row["gold_label"]}, ensure_ascii=False) + "\n")
    logger.info(f"JSONL saved: {path}  ({df.height} records)")


def main() -> None:
    input_path = DATA_PROCESSED / INPUT_FILE
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run src/inference.py first.")
        return

    logger.info(f"Loading {input_path} ...")
    df = pl.read_parquet(input_path)
    logger.info(f"Loaded {df.height:,} rows.")

    sample  = build_sample(df)
    labeled = label_sample(sample)

    labeled.write_parquet(DATA_PROCESSED / OUTPUT_PARQUET)
    logger.info(f"Parquet saved: {DATA_PROCESSED / OUTPUT_PARQUET}")

    save_jsonl(labeled, DATA_PROCESSED / OUTPUT_JSONL)

    agreed = (labeled["label"] == labeled["gold_label"]).sum()
    logger.info(f"Silver→Gold agreement: {agreed}/{labeled.height} ({agreed/labeled.height:.1%})")


if __name__ == "__main__":
    main()
