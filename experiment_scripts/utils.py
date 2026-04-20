import unicodedata, os, torch, random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import precision_recall_fscore_support
from transformers import EarlyStoppingCallback

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

from torch.utils.data import DataLoader


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
# TRAINING FUNCTION
# =========================

def run_training(seed, lr, MODEL_NAME, args, train_dataset, val_dataset, data_collator):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, "tmp"),
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=lr,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        max_steps=10000,
        load_best_model_at_end=True,
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    results = trainer.evaluate()

    best_steps = trainer.state.best_global_step

    del trainer
    del model

    torch.cuda.empty_cache()

    return results["eval_f1_macro"], best_steps

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

def is_punctuation(ch: str) -> bool:
    """
    Returns True if the input character is any type of Unicode punctuation.
    """
    if not ch or len(ch) != 1:
        raise ValueError("Input must be a single character.")
    
    # Unicode punctuation categories start with "P"
    return unicodedata.category(ch).startswith("P")

# def is_dutch(text):
#     label, probability = model.predict(text)
#     if label == '__label__nl' and probability[0] > 0.5:  # Adjust threshold as needed
#         return True
#     else:
#         return False
    
def preprocess(text):
    # remove_punctuation
    text = ''.join(ch for ch in text if not is_punctuation(ch))

    # remove think tokens
    text = text.replace("think", "")

    return text

def plot_svm_top_features_per_class(clf_pipeline, feature_names, output_dir, top_n=20):
    """
    Extract top N SVM features per class from a Pipeline containing:
        - A TfidfVectorizer step named "vec"
        - A LinearSVC/SVC(kernel="linear") step named "svm"

    Handles both binary and multiclass SVMs correctly.

    Saves an HTML file with Plotly visualizations.

    Parameters
    ----------
    clf_pipeline : sklearn Pipeline
        Must contain steps "vec" and "svm"
    feature_names : list of str
        Names of TF-IDF features
    output_dir : str
        Directory where HTML will be saved
    top_n : int
        Number of top features per class
    """

    # ---------------------------------------------------------
    # 1. Extract classifier
    # ---------------------------------------------------------
    clf = clf_pipeline["svm"]

    if not hasattr(clf, "coef_"):
        raise ValueError("Classifier must be a linear SVM with a coef_ attribute.")

    coefs = clf.coef_
    classes = clf.classes_
    n_classes = len(classes)

    # ---------------------------------------------------------
    # 2. Handle binary SVM case: coef_.shape = (1, n_features)
    # ---------------------------------------------------------
    if coefs.shape[0] == 1 and n_classes == 2:
        coef_matrix = np.vstack([coefs[0], -coefs[0]])
    else:
        coef_matrix = coefs

    # ---------------------------------------------------------
    # 3. Plotting setup - share x-axis for compactness
    # ---------------------------------------------------------
    fig = make_subplots(
        rows=n_classes,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Top {top_n} Features for Class: {cls}" for cls in classes],
        vertical_spacing=0.05
    )

    # ---------------------------------------------------------
    # 4. Build one subplot per class
    # ---------------------------------------------------------
    for idx, cls in enumerate(classes):
        class_coefs = coef_matrix[idx]

        top_indices = np.argsort(np.abs(class_coefs))[-top_n:]
        top_indices = top_indices[np.argsort(class_coefs[top_indices])]

        top_features = [feature_names[i] for i in top_indices]
        top_weights = class_coefs[top_indices]

        fig.add_trace(
            go.Bar(
                x=top_weights,
                y=top_features,
                orientation="h",
                name=str(cls),
                marker=dict(line=dict(width=1)),  # slightly thicker bar edges
                width=0.6  # thicker bars
            ),
            row=idx + 1,
            col=1
        )

        # Force all ticks to show for each subplot
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
            tickfont=dict(size=10),
            automargin=True,
            row=idx + 1,
            col=1
        )

    # ---------------------------------------------------------
    # 5. Layout + save
    # ---------------------------------------------------------
    fig.update_layout(
        height=250 * n_classes,
        title="Top SVM Features per Class",
        showlegend=False,
        margin=dict(l=220, r=50, t=100, b=50)  # increase left margin for long names
    )

    output_path = os.path.join(output_dir, "svm_top_features.html")
    fig.write_html(output_path)

    return fig


def get_pipeline_feature_names(pipeline):
    """
    Extract the full feature name list from the ColumnTransformer inside the pipeline.
    Works for TfidfVectorizer, CountVectorizer, etc.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    for name, transformer, column in preprocessor.transformers_:

        # Skip dropped transformers
        if transformer == "drop":
            continue

        # Case 1: Standard vectorizers (TfidfVectorizer, CountVectorizer)
        if hasattr(transformer, "get_feature_names_out"):
            fn = transformer.get_feature_names_out()
            feature_names.extend([f"{name}__{feat}" for feat in fn])
            continue

        # Case 2: Pipelines inside ColumnTransformer (not used here, but safe)
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                fn = last_step.get_feature_names_out()
                feature_names.extend([f"{name}__{feat}" for feat in fn])
                continue

        # Unexpected transformer type
        raise ValueError(f"Cannot extract features from transformer: {name}")

    return feature_names

def plot_rf_top_features(
    clf_pipeline,
    feature_names,
    output_dir,
    top_n=20
):
    """
    Extract top N features from a Pipeline containing:
        - A TfidfVectorizer step named "vec"
        - A RandomForestClassifier step named "rf"

    Uses feature_importances_.

    Saves an HTML file with Plotly visualization.
    """

    # ---------------------------------------------------------
    # 1. Extract classifier
    # ---------------------------------------------------------
    clf = clf_pipeline["rf"]

    if not hasattr(clf, "feature_importances_"):
        raise ValueError("Classifier must have feature_importances_ attribute.")

    importances = clf.feature_importances_

    # ---------------------------------------------------------
    # 2. Select top features
    # ---------------------------------------------------------
    top_indices = np.argsort(importances)[-top_n:]
    top_indices = top_indices[np.argsort(importances[top_indices])]

    top_features = [feature_names[i] for i in top_indices]
    top_values = importances[top_indices]

    # ---------------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_values,
            y=top_features,
            orientation="h",
            marker=dict(line=dict(width=1)),
            width=0.6
        )
    )

    fig.update_layout(
        height=500,
        title=f"Top {top_n} Random Forest Features (Global Importance)",
        showlegend=False,
        margin=dict(l=220, r=50, t=100, b=50)
    )

    # ---------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------
    output_path = os.path.join(output_dir, "rf_top_features.html")
    fig.write_html(output_path)

    return fig