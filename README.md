# ⚖️ Political Discourse Analysis: Brazil 2022

> **Project: Auditing Democracy**
> An end-to-end NLP pipeline to audit the 2022 Brazilian presidential election discourse, distinguishing legitimate political combativeness from antidemocratic rhetoric.

## Objective

The 2022 Brazilian presidential election was marked by unprecedented polarization. Standard sentiment analysis fails to address the core methodological problem: both legitimate political opposition and genuinely antidemocratic rhetoric are "negative" in tone, but only one represents a systemic threat to institutions. This project builds a classification pipeline with sufficient granularity to make that distinction at scale.

## Taxonomy

| Label | Description |
|---|---|
| 🔴 retórica antidemocrática | Authorial, direct threats to democratic institutions |
| 🟠 acusação antidemocrática | Accusations against the opponent of antidemocratic behavior |
| ⚪ ataques políticos | Personal attacks, insults, moral or competence criticism |
| 🔵 administração pública | Government actions, policy, legislation, public management |
| 🟡 política econômica | Economy-focused content: inflation, employment, fiscal policy |
| 🟢 campanha eleitoral | Electoral mobilization, polling, campaign events |
| ⬜ neutro | Informational, institutional, or uncategorizable content |

## Pipeline Architecture

### 1. Ingestion — `src/political_collector.py`

Fetches the complete message history from the official Telegram channels (`@jairbolsonarobrasil`, `@LulanoTelegram`) using Telethon, covering January 2018 to January 2023. Outputs a sorted, cleaned parquet file to `data/raw/`.

### 2. Hybrid Classification — `src/inference.py`

Classifies Portuguese text natively using a two-layer architecture:

**XLM-R Zero-Shot** (`joeddav/xlm-roberta-large-xnli`): a multilingual transformer capable of zero-shot classification without any labeled data, using a Portuguese-language hypothesis template. Classifying on the original language eliminates the accuracy degradation inherent to pre-classification translation.

**Regex Heuristic Layer**: a deterministic guardrail for high-stakes edge cases. The key challenge it resolves is the authorship ambiguity: both *"Vamos invadir o STF"* (systemic rhetoric) and *"O Bolsonaro quer invadir o STF"* (accusation) contain identical threat-level vocabulary. The heuristic detects citation verbs and negation patterns to determine whether the channel is authoring or reporting a claim, then overrides the neural prediction accordingly.

After fine-tuning, the inference module automatically detects the trained model at `models/bert-pt-political/` and replaces the zero-shot bootstrap with it.

### 3. LLM Annotation — `src/labeler.py`

Generates a gold-standard fine-tuning dataset via a stratified sampling strategy:

- All posts classified as systemic rhetoric or antidemocratic accusation (priority classes, capped at 200 each)
- All low-confidence predictions (score < 0.65), where the zero-shot model was most uncertain
- A balanced draw of 50 posts per remaining label

Each sampled post is sent to `claude-opus-4-7` with a cached domain-specific system prompt for expert-level relabeling. Prompt caching reduces token cost by approximately 90% across the batch. Outputs `finetuning_sample.parquet` and `finetuning_sample.jsonl`.

### 4. Fine-Tuning — `src/finetuner.py`

Fine-tunes `neuralmind/bert-base-portuguese-cased` (BERTimbau) on the Claude-labeled sample:

- Stratified 80/20 train/validation split
- Inverse-frequency class weighting via a custom `WeightedTrainer` to correct label imbalance introduced by the stratified sampling strategy
- Early stopping on macro F1 (patience = 2 epochs)
- Model and tokenizer persisted to `models/bert-pt-political/`

Upon completion, `src/inference.py` automatically uses the fine-tuned model on the next run.

**Corpus Finding — `retórica antidemocrática`.** Manual validation of the full corpus confirmed that the official Telegram channels contain virtually no examples of direct authorial antidemocratic rhetoric as strictly defined. A systematic search covering epistemic attacks on the electoral system, conditional acceptance of results, military mobilization discourse, and judicial delegitimization returned a single non-ambiguous candidate from 6,950 posts. This is methodologically significant: the official Telegram channel functioned primarily as a governance and campaign communication platform; the most explicit antidemocratic discourse occurred in live broadcasts, rallies, and direct social media interactions not captured in this corpus. As a consequence, the fine-tuned model covers six of the seven taxonomy labels; the `retórica antidemocrática` class is handled exclusively by the deterministic heuristic layer in `src/inference.py`.

### 5. Display Translation — `src/translator_gold.py`

Translates the classified Portuguese corpus to English using `Helsinki-NLP/opus-mt-roa-en`, exclusively for front-end display. Long posts are split into sentence-level chunks before translation and reassembled. Classification is performed on the original Portuguese text to maximize accuracy.

### 6. ADI Index — `src/indexer.py`

Computes the **Antidemocratic Discourse Index (ADI)** per candidate over time across two components:

$$ADI^{acc}_{monthly}(c,t) = \frac{\sum_{i:\, l_i = \text{acusação antidemocrática}} score_i}{N(c,t)} \times 100$$

$$ADI^{att}_{monthly}(c,t) = \frac{\sum_{i:\, l_i = \text{ataques políticos}} score_i}{N(c,t)} \times 100$$

$$ADI^{k}_{cumulative}(c,t) = \sum_{s \leq t} ADI^{k}_{monthly}(c,s)$$

Where $score_i$ is the model's confidence for post $i$ and $N(c,t)$ is the total number of posts by candidate $c$ in month $t$. The index tracks antidemocratic accusations (attributing institutional threats to the opponent) and political attacks (personal, moral, and competence-based aggression) as separate components, allowing the temporal evolution of each discourse type to be analysed independently. Normalizing by volume prevents high-activity months from inflating the index; weighting by confidence means uncertain predictions contribute proportionally less. Outputs `data/processed/adi_index.parquet`.

### 7. Dashboard — `app.py`

Interactive Streamlit dashboard with three views:

- **Dashboard Overview**: temporal trend analysis of rhetoric and accusation intensity per candidate, combining posting volume bars with percentage-of-month line traces on a dual-axis chart
- **Data Explorer**: filterable dataframe exposing raw labels, confidence scores, and translated content
- **Semantic Cloud**: word cloud segmented by category and candidate, rendered on the translated corpus
- **Antidemocratic Index**: dual-axis chart of monthly ADI intensity (bars) and cumulative load (lines) per candidate, plus exportable monthly table

---

## Tech Stack

Python 3.10 · Polars · PyTorch · Hugging Face Transformers · XLM-R (`joeddav/xlm-roberta-large-xnli`) · BERTimbau (`neuralmind/bert-base-portuguese-cased`) · Helsinki-NLP (`opus-mt-roa-en`) · Anthropic API (`claude-opus-4-7`) · Streamlit · Plotly

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conda environment `nlp_politics` (PyTorch 2.3.1 + CUDA 12.1)
- Telegram API credentials (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`)
- Anthropic API key (`ANTHROPIC_API_KEY`)

### Setup

```bash
conda activate nlp_politics
cd C:\Users\luisg\projetos\political-discourse-analysis
copy .env.example .env   # fill in credentials
```

### Execution

```bash
python -m src.political_collector   # scrape Telegram → data/raw/
python -m src.inference             # classify (XLM-R + heuristics) → labeled_discourse_pt.parquet
python -m src.translator_gold       # translate for display → labeled_v4_discourse.parquet
python -m src.labeler               # annotate with Claude → finetuning_sample.jsonl
python -m src.finetuner             # fine-tune BERTimbau → models/bert-pt-political/
python -m src.inference             # reclassify with fine-tuned model
python -m src.indexer               # compute ADI → adi_index.parquet
streamlit run app.py                # launch dashboard
```
