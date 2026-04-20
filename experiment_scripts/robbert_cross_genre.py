import os, argparse, wandb, torch, utils
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # PyTorch >=1.8

MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"

# =========================
# GLOBAL DETERMINISM SETUP
# =========================

os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"

# =========================
# PARSE ARGUMENTS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# =========================
# LOAD DATA
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
# SET HYPERPARAMETERS
# =========================

learning_rates = [1e-4, 5e-5, 1e-5]
seeds = [42, 43, 44, 45, 46]
# steps will be optimized base on early stopping on validation set
# batch size is set to maximum that fits in GPU memory for faster training (64)

# =========================
# STORE DATA STATISTICS
# ========================
df_genre_stats = pd.DataFrame()
data = {
    'genre': [], 

    'n_total': [], 
    'n_human': [], 
    'n_ai': [], 

    'ratio_human': [], 
    'ratio_ai': [],
}


# =========================
# BEGIN CROSS-GENRE EVALUATION
# ========================
for genre in tqdm(df["genre"].unique()):
    print("\n===OUT-OF-DOMAIN GENRE:", genre, "===\n")

    # DATA SPLIT
    train_val_df, test_df = df[df["genre"] != genre], df[df["genre"] == genre]

    # To ensure a representative validation set, we stratify on both label and genre
    stratify_cols = train_val_df[["binary_label", "genre"]]

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,
        stratify=stratify_cols.apply(tuple, axis=1),
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_val_dataset = Dataset.from_pandas(train_val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # fast batched tokenization
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=1000)
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=1000)
    train_val_dataset = train_val_dataset.map(tokenize, batched=True, batch_size=1000)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=1000)

    train_dataset = train_dataset.rename_column("binary_label", "labels")
    val_dataset = val_dataset.rename_column("binary_label", "labels")
    train_val_dataset = train_val_dataset.rename_column("binary_label", "labels")
    test_dataset = test_dataset.rename_column("binary_label", "labels")

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

    test_dataset.set_format(
        type="torch",
        columns=["input_ids","attention_mask","labels"]
    )

    # save statistics
    n_total = len(test_df)
    n_human = len(test_df[test_df["binary_label"] == 0])
    n_ai = len(test_df[test_df["binary_label"] == 1])
    ratio_human = n_human / n_total if n_total > 0 else 0
    ratio_ai = n_ai / n_total if n_total > 0 else 0

    data['genre'].append(genre)
    data['n_total'].append(n_total)
    data['n_human'].append(n_human)
    data['n_ai'].append(n_ai)
    data['ratio_human'].append(ratio_human)
    data['ratio_ai'].append(ratio_ai)

    # =========================
    # GRID SEARCH
    # =========================

    best_configs = {}

    for seed in seeds:

        best_f1 = -1
        best_params = None

        for lr in learning_rates:

            run_name = f"Test{genre}_seed{seed}_lr{lr}"

            wandb.init(
                project="robbert-ai-detection-cross-genre",
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

        with open(os.path.join(args.output, f"{genre}_best_params_seed{seed}.txt"), "w") as f:

            f.write(f"seed: {seed}\n")
            f.write(f"learning_rate: {best_params['learning_rate']}\n")
            f.write(f"num_steps: {best_params['num_steps']}\n")

        # ============================
        # FINAL TRAINING ON TRAIN+VAL 
        # AND EVALUATION ON TEST SET
        # ============================

        run_name = f"{genre}_inference_seed_{seed}"

        wandb.init(
            project="robbert-ai-detection-cross-genre",
            name=run_name,
            config={
                "seed": seed,
                "learning_rate": best_params["learning_rate"],
                "batch_size": 64,
                "stage": "test_evaluation"
            }
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output, "tmp_final"),
            eval_strategy="no",
            save_strategy="no",
            learning_rate=best_params["learning_rate"],
            per_device_train_batch_size=64,
            max_steps=best_params["num_steps"],
            logging_strategy="steps",
            report_to="wandb",
            seed=seed,
        )

        trainer = utils.DeterministicTrainer(
            model=model,
            args=training_args,
            train_dataset=train_val_dataset,
            data_collator=data_collator
        )

        trainer.train()

        # FINAL EVALUATION ON TEST SET
        test_preds = trainer.predict(test_dataset)
        preds = np.argmax(test_preds.predictions, axis=1)

        metrics = utils.compute_metrics(
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
            os.path.join(args.output, f"{genre}_test_predictions_seed{seed}.csv"),
            index=False
        )

        wandb.finish()

        del trainer
        del model

        torch.cuda.empty_cache()