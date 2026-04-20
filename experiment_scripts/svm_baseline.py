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
# Train–test split
# -------------------------
train_split, test_split = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["binary_label"]
)

X_train = train_split["text"]
y_train = train_split["binary_label"]
z_train = train_split["genre"]

X_test = test_split["text"]
y_test = test_split["binary_label"]
z_test = test_split["genre"]

# -------------------------
# Pipeline
# -------------------------
pipeline = Pipeline([
    ("vec", TfidfVectorizer()),
    ("svm", LinearSVC(random_state=42, class_weight="balanced")),
])

param_grid = {
    "vec__lowercase": [True], #True, False
    "vec__max_df": [1.0], #1.0, 0.9, 0.8
    "vec__min_df": [5], #1, 5, 10
    "vec__ngram_range": [(1,2)], #(1,1), (1,2), (1,3)
    "svm__C": [0.1], #0.1, 1, 10
    "svm__class_weight": [{0: 1.0, 1: 2.0}] #None, balanced, 
}

# -------------------------
# Cross-validation
# -------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=5,
    verbose=1
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
print(best_params_df)

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
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_f, y_pred_f
    )
    metrics.append({
        "Fold": fold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    })
    # save all predictions w. genre
    fold_predictions_df = pd.DataFrame({
        "text": X_train.iloc[test_idx],
        "true_label": y_true_f,
        "predicted_label": y_pred_f,
        "genre": z_train.iloc[test_idx]
    })
    fold_predictions_df.to_csv(os.path.join(args.output, f"fold_{fold}_predictions.csv"), index=False)

# -------------------------
# Test set
# -------------------------
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

report_df = pd.DataFrame(
    classification_report(
        y_test, y_test_pred, 
        output_dict=True,
        target_names=["human", "AI-generated"],
        digits=3,
    )
).transpose()

report_df.to_csv(os.path.join(args.output, "test_classification_report.csv"))
print(report_df)

#--------------------------
# Save test set predictions
#--------------------------
test_predictions_df = pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "predicted_label": y_test_pred,
    "genre": z_test
})
test_predictions_df.to_csv(os.path.join(args.output, "test_set_predictions.csv"), index=False)

# -------------------------
# Feature analysis
# -------------------------
vectorizer = best_model.named_steps["vec"]
feature_names = vectorizer.get_feature_names_out()
utils.plot_svm_top_features_per_class(best_model, feature_names, args.output)

#------CALIBRATION-------

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

calibrated_pipeline = Pipeline([
    ("vec", best_pipeline.named_steps["vec"]),
    (
        "svm",
        CalibratedClassifierCV(
            estimator=best_pipeline.named_steps["svm"],
            method="sigmoid",
            cv=5
        )
    )
])

calibrated_pipeline.fit(X_train, y_train)

#-------SAVE_MODEL--------

import joblib
import os

model_path = os.path.join(args.output, "calibrated_svm_pipeline.joblib")
joblib.dump(calibrated_pipeline, model_path)

print(f"Model saved to {model_path}")