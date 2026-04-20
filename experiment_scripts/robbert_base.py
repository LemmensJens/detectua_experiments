import os
import argparse
import random
import wandb
import torch
import utils
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

from torch.utils.data import DataLoader


# =========================
# GLOBAL DETERMINISM SETUP
# =========================

os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"


# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)


# =========================
# DATA
# =========================

df = pd.read_csv(args.input)

label_dist = df["binary_label"].value_counts()
label_dist.to_csv(
    os.path.join(args.output, "binary_label_distribution.csv"),
    header=["count"]
)

df["text_length"] = df["text"].apply(lambda x: len(x.split()))
text_length_dist = df["text_length"].value_counts().sort_index()

text_length_dist.to_csv(
    os.path.join(args.output, "text_length_distribution.csv"),
    header=["count"]
)


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0,1],
        zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "precision_human": precision[0],
        "recall_human": recall[0],
        "f1_human": f1[0],
        "precision_ai": precision[1],
        "recall_ai": recall[1],
        "f1_ai": f1[1],
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1
    }


# =========================
# TOKENIZER 
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512
    )

data_collator = DataCollatorWithPadding(tokenizer)


# =========================
# DATA SPLIT
# =========================

train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["binary_label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1,
    stratify=train_val_df["binary_label"],
    random_state=42
)


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
train_val_dataset = Dataset.from_pandas(train_val_df)


# fast batched tokenization

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=1000)
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=1000)
train_val_dataset = train_val_dataset.map(tokenize, batched=True, batch_size=1000)


train_dataset = train_dataset.rename_column("binary_label", "labels")
val_dataset = val_dataset.rename_column("binary_label", "labels")
train_val_dataset = train_val_dataset.rename_column("binary_label", "labels")


train_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)

train_val_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)


# =========================
# DETERMINISTIC DATALOADER
# =========================

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DeterministicTrainer(Trainer):

    def get_train_dataloader(self):

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator
        )


# =========================
# HYPERPARAMETERS
# =========================

learning_rates = [1e-4, 5e-5, 1e-5]

seeds = [42, 43, 44, 45, 46]


# =========================
# GRID SEARCH
# =========================

best_configs = {}

for seed in seeds:

    best_f1 = -1
    best_params = None

    for lr in learning_rates:

        run_name = f"search_seed{seed}_lr{lr}"

        wandb.init(
            project="robbert-ai-detection",
            name=run_name,
            config={
                "seed": seed,
                "learning_rate": lr
            }
        )

        f1, steps = utils.run_training(seed, lr, MODEL_NAME, args, train_dataset, val_dataset, data_collator)

        if f1 > best_f1:
            best_f1 = f1
            best_params = {"learning_rate": lr, "num_steps": steps}

        wandb.finish()

    best_configs[seed] = best_params

    with open(os.path.join(args.output, f"best_params_seed{seed}.txt"), "w") as f:

        f.write(f"seed: {seed}\n")
        f.write(f"learning_rate: {best_params['learning_rate']}\n")
        f.write(f"num_steps: {best_params['num_steps']}\n")


# =========================
# TEST DATASET
# =========================

test_dataset = Dataset.from_pandas(test_df)

test_dataset = test_dataset.map(tokenize, batched=True, batch_size=1000)

test_dataset = test_dataset.rename_column("binary_label", "labels")

test_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)


# =========================
# FINAL TRAINING
# =========================

for seed in seeds:

    params = best_configs[seed]

    wandb.init(
        project="robbert-ai-detection",
        name=f"inference_seed_{seed}"
    )

    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, "tmp_final"),
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=64,
        max_steps=int(params["num_steps"]),
        num_train_epochs=0,
        logging_strategy="steps",
        report_to="wandb",
        seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=True
    )

    trainer = DeterministicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    test_preds = trainer.predict(test_dataset)

    preds = np.argmax(test_preds.predictions, axis=1)

    metrics = compute_metrics(
        (test_preds.predictions, test_df["binary_label"].values)
    )

    wandb.log(metrics)

    test_out = pd.DataFrame({
        "text": test_df["text"],
        "true_label": test_df["binary_label"],
        "predicted_label": preds,
        "genre": test_df["genre"]
    })

    test_out.to_csv(
        os.path.join(args.output, f"test_predictions_seed{seed}.csv"),
        index=False
    )

    wandb.finish()

    del trainer
    del model

    torch.cuda.empty_cache()