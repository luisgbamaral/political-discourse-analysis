import json
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.utils import BASE_DIR, DATA_PROCESSED, get_logger

logger = get_logger("FineTuner")

MODEL_DIR  = BASE_DIR / "models" / "bert-large-pt-political"
JSONL_PATH = DATA_PROCESSED / "finetuning_sample_v4.jsonl"
BASE_MODEL = "neuralmind/bert-large-portuguese-cased"

LABELS = [
    "retórica antidemocrática",
    "acusação antidemocrática",
    "ataques políticos",
    "administração pública",
    "política econômica",
    "campanha eleitoral",
    "neutro",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

MAX_LEN    = 256
BATCH_SIZE = 8       # bert-large needs ~4× more VRAM than base; halve batch and accumulate
GRAD_ACCUM = 2       # effective batch = 8 × 2 = 16 (same as before)
EPOCHS     = 5
LR         = 1e-5    # lower LR for larger model


class PoliticalDataset(Dataset):
    def __init__(self, texts: list[str], label_ids: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    texts, label_ids = [], []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            label = row.get("label", "").strip().lower()
            if label not in LABEL2ID:
                skipped += 1
                continue
            texts.append(row["text"])
            label_ids.append(LABEL2ID[label])
    if skipped:
        logger.warning(f"Skipped {skipped} rows with unrecognised labels.")
    return texts, label_ids


def main() -> None:
    if not JSONL_PATH.exists():
        logger.error(f"Fine-tuning data not found: {JSONL_PATH}. Run src/labeler.py first.")
        return

    logger.info(f"Loading {JSONL_PATH} ...")
    texts, label_ids = load_jsonl(JSONL_PATH)
    logger.info(f"Loaded {len(texts)} labeled examples")

    if len(texts) < 10:
        logger.error("Too few examples to train. Check finetuning_sample.jsonl.")
        return

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, label_ids,
        test_size=0.2,
        random_state=42,
        stratify=label_ids,
    )
    logger.info(f"Train: {len(tr_texts)}  |  Val: {len(val_texts)}")

    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = PoliticalDataset(tr_texts, tr_labels, tokenizer, MAX_LEN)
    val_ds   = PoliticalDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    present = np.unique(tr_labels)
    partial = compute_class_weight(class_weight="balanced", classes=present, y=np.array(tr_labels))
    weights = np.ones(len(LABELS))
    for cls, w in zip(present, partial):
        weights[cls] = w
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info("Class weights: " + str({LABELS[i]: round(float(w), 3) for i, w in enumerate(weights)}))

    logger.info(f"Loading model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Training...")
    trainer.train()

    results = trainer.evaluate()
    logger.info(f"Best val F1 (macro): {results['eval_f1_macro']:.4f}")

    preds_out = trainer.predict(val_ds)
    val_preds = np.argmax(preds_out.predictions, axis=-1)
    report = classification_report(val_labels, val_preds, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    logger.info(f"Model saved → {MODEL_DIR}")


if __name__ == "__main__":
    main()
