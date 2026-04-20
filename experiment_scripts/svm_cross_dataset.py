import os
import argparse
import pandas as pd
from tqdm import tqdm
import stanza
import utils
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline as MiniPipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV


tqdm.pandas()

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(args.input)

# compute and save label distribution
label_dist = df["binary_label"].value_counts()
label_dist.to_csv(os.path.join(args.output, "binary_label_distribution.csv"), header=["count"])

# compute and save frequency of text lengths
df["text_length"] = df["text"].apply(lambda x: len(x.split()))
text_length_dist = df["text_length"].value_counts().sort_index()
text_length_dist.to_csv(os.path.join(args.output, "text_length_distribution.csv"), header=["count"])

# -------------------------
# Train–test splits
# -------------------------
df_dataset_stats = pd.DataFrame()
data = {
    'dataset': [], 

    'n_total': [], 
    'n_human': [], 
    'n_ai': [], 

    'ratio_human': [], 
    'ratio_ai': [],

    'pre_human': [],
    'rec_human': [],
    'f1_human': [],

    'pre_ai': [],
    'rec_ai': [],
    'f1_ai': [],

    'pre_macro': [],
    'rec_macro': [],
    'f1_macro': [],
}

for dataset in tqdm(df["corpus"].unique()):
    print("\n===OUT-OF-DOMAIN DATASET:", dataset, "===\n")

    test_split = df[df["corpus"] == dataset]
    train_split = df[df["corpus"] != dataset]

    X_train = train_split["text"]
    y_train = train_split["binary_label"]

    X_test = test_split["text"]
    y_test = test_split["binary_label"]

    # save statistics
    n_total = len(test_split)
    n_human = len(test_split[test_split["binary_label"] == 0])
    n_ai = len(test_split[test_split["binary_label"] == 1])
    ratio_human = n_human / n_total if n_total > 0 else 0
    ratio_ai = n_ai / n_total if n_total > 0 else 0

    data['dataset'].append(dataset)
    data['n_total'].append(n_total)
    data['n_human'].append(n_human)
    data['n_ai'].append(n_ai)
    data['ratio_human'].append(ratio_human)
    data['ratio_ai'].append(ratio_ai)

    # -------------------------
    # Pipeline
    # -------------------------
    pipeline = Pipeline([
        ("vec", TfidfVectorizer()),
        ("svm", LinearSVC(random_state=42)),
    ])

    param_grid = {
        "vec__lowercase": [True, False],
        "vec__max_df": [1.0, 0.9, 0.8], 
        "vec__min_df": [1, 2, 5, 10], 
        "vec__ngram_range": [(1, 1), (1, 2)],
        "svm__C": [0.1, 1, 2, 3, 5], 
        "svm__class_weight": [None, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 1.5}, {0: 1.0, 1: 2.5}],
    }

    # -------------------------
    # Cross-validation
    # -------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=5,
        verbose=10
    )
    grid.fit(X_train, y_train)

    # -------------------------
    # Fit
    # -------------------------
    grid.fit(X_train, y_train)

    best_params_df = pd.DataFrame(
        list(grid.best_params_.items()),
        columns=["Parameter", "Best Value"]
    )

    # -------------------------
    # CV predictions
    # -------------------------
    best_pipeline = grid.best_estimator_

    y_pred = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv
    )

    metrics = []
    for fold, (_, test_idx) in enumerate(cv.split(X_train, y_train), 1):
        y_true_f = y_train.iloc[test_idx]
        y_pred_f = y_pred[test_idx]

        # Results on human class
        p_h, r_h, f_h, s_h = precision_recall_fscore_support(y_true_f, y_pred_f, average="binary", pos_label=0)

        # Results on AI class
        p_ai, r_ai, f_ai, s_ai = precision_recall_fscore_support(y_true_f, y_pred_f, average="binary")

        # Macro-averaged results
        p_macro, r_macro, f_macro, s_macro = precision_recall_fscore_support(y_true_f, y_pred_f, average="macro")

        metrics.append({
            "fold": fold,

            'pre_human': round(p_h, 3),
            'rec_human': round(r_h, 3),
            'f1_human': round(f_h, 3),

            'pre_ai': round(p_ai, 3),
            'rec_ai': round(r_ai, 3),
            'f1_ai': round(f_ai, 3),

            'pre_macro': round(p_macro, 3),
            'rec_macro': round(r_macro, 3),
            'f1_macro': round(f_macro, 3),
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.loc[len(metrics_df)] = {
        "fold": "Unweighted mean ± std.",
        
        "pre_human": f"{metrics_df['pre_human'].mean():.3f} ± {metrics_df['pre_human'].std():.3f}",
        "rec_human": f"{metrics_df['rec_human'].mean():.3f} ± {metrics_df['rec_human'].std():.3f}",
        "f1_human": f"{metrics_df['f1_human'].mean():.3f} ± {metrics_df['f1_human'].std():.3f}",
        
        "pre_ai": f"{metrics_df['pre_ai'].mean():.3f} ± {metrics_df['pre_ai'].std():.3f}",
        "rec_ai": f"{metrics_df['rec_ai'].mean():.3f} ± {metrics_df['rec_ai'].std():.3f}",
        "f1_ai": f"{metrics_df['f1_ai'].mean():.3f} ± {metrics_df['f1_ai'].std():.3f}",
        
        "pre_macro": f"{metrics_df['pre_macro'].mean():.3f} ± {metrics_df['pre_macro'].std():.3f}",
        "rec_macro": f"{metrics_df['rec_macro'].mean():.3f} ± {metrics_df['rec_macro'].std():.3f}",
        "f1_macro": f"{metrics_df['f1_macro'].mean():.3f} ± {metrics_df['f1_macro'].std():.3f}",
    }

    metrics_df.to_csv(os.path.join(args.output, f"cv_results_no_{dataset}.csv"), index=False)

    # -------------------------
    # Test set predictions
    # -------------------------
    best_model = grid.best_estimator_
    y_test_pred = best_model.predict(X_test)
    with open(f"{args.output}/best_params_{dataset}.txt", "w") as f:
        for key, value in grid.best_params_.items():
            f.write(f"{key}: {value}\n")

    # Results on human class
    p_h, r_h, f_h, s_h = precision_recall_fscore_support(y_test, y_test_pred, average="binary")

    # Results on AI class
    p_ai, r_ai, f_ai, s_ai = precision_recall_fscore_support(y_test, y_test_pred, average="binary")

    # Macro-averaged results
    p_macro, r_macro, f_macro, s_macro = precision_recall_fscore_support(y_test, y_test_pred, average="macro")

    # Add results to data dict
    data['pre_human'].append(round(p_h, 3))
    data['rec_human'].append(round(r_h, 3))
    data['f1_human'].append(round(f_h, 3))

    data['pre_ai'].append(round(p_ai, 3))
    data['rec_ai'].append(round(r_ai, 3))
    data['f1_ai'].append(round(f_ai, 3))

    data['pre_macro'].append(round(p_macro, 3))
    data['rec_macro'].append(round(r_macro, 3))
    data['f1_macro'].append(round(f_macro, 3))

    # Save dataset predictions to CSV
    test_results_df = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "predicted_label": y_test_pred,
        "genre": test_split["genre"],
    })
    test_results_df.to_csv(os.path.join(args.output, f"test_predictions_{dataset}.csv"), index=False)

# Save results to CSV
results_df = pd.DataFrame(data)
mean_row = results_df.mean(numeric_only=True).round(3)
mean_row = mean_row.reindex(results_df.columns, fill_value='Unweighted mean')
results_df.loc[len(results_df)] = mean_row

results_df.to_csv(os.path.join(args.output, "cross_dataset_classification_report.csv"), index=False)