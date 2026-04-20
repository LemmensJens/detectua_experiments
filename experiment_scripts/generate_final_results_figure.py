import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Data ----
data = {
    'Metric': ['Precision', 'Recall', 'F1-score'],
    'clin_svm': [83.6, 67.5, 74.6],
    'clin_rf': [90.4, 55.1, 68.5],
    'clin_robbert': [75.5, 67.9, 71.4],
    'news_svm': [97.1, 99.8, 98.5],
    'news_rf': [99.1, 96.5, 97.8],
    'news_robbert': [100.0, 98.7, 99.4],
    'csi_svm': [87.8, 100.0, 93.5],
    'csi_rf': [95.6, 94.2, 94.9],
    'csi_robbert': [87.6, 100.0, 93.4],
}

df = pd.DataFrame(data)
df_long = df.melt(id_vars="Metric", var_name="Model", value_name="Score")
df_long[['Dataset', 'Model_Type']] = df_long['Model'].str.split('_', expand=True)
df_long['Model_Type'] = df_long['Model_Type'].str.upper()

# ---- Load Info ----
load_data = {
    "Model": ["SVM", "RF", "ROBBERT"],
    "Memory / Load": [
        "Idle: 0.02MB; Active: 177MB (CPU)",
        "Similar/slightly larger than SVM",
        "Idle: At least 500MB; Active: ~2GB (GPU)"
    ],
    "Speed": [
        "60s / 10k docs",
        "Slightly slower than SVM",
        "125s / 10k docs (excl. tokenization)"
    ]
}

# ---- Create Subplots ----
datasets = df_long["Dataset"].unique()
n_datasets = len(datasets)

fig = make_subplots(
    rows=2, cols=n_datasets,
    row_heights=[0.7, 0.3],
    specs=[[{"type": "xy"}]*n_datasets,
           [{"type": "table", "colspan": n_datasets}]+[None]*(n_datasets-1)],
    vertical_spacing=0.08,
    subplot_titles=[d.upper() for d in datasets],
    shared_yaxes=True
)

# ---- Add Bar Charts per Dataset ----
colors = {"Precision": "#636EFA", "Recall": "#EF553B", "F1-score": "#00CC96"}

for i, dataset in enumerate(datasets):
    subset = df_long[df_long["Dataset"] == dataset]
    for metric in subset["Metric"].unique():
        metric_subset = subset[subset["Metric"] == metric]
        fig.add_trace(
            go.Bar(
                x=metric_subset["Model_Type"],
                y=metric_subset["Score"],
                name=metric if i==0 else None,  # show legend only once
                marker_color=colors[metric],
                text=metric_subset["Score"],
                textposition="auto",
                showlegend=(i==0)  # only show legend for first dataset
            ),
            row=1, col=i+1
        )

# ---- Add Table in bottom row (spanning all columns) ----
fig.add_trace(
    go.Table(
        header=dict(values=list(load_data.keys()), fill_color='lightgrey', align='center'),
        cells=dict(values=list(load_data.values()), align='left')
    ),
    row=2, col=1
)

# ---- Layout ----
fig.update_layout(
    height=700,
    width=1100,
    title_text="Model Performance per Dataset with Resource Load",
    title_x=0.5,
    barmode="group",
    showlegend=True,
    legend=dict(
        title="Metric",
        orientation="h",
        y=0.275,      
        x=0.5,
        xanchor="center",
        yanchor="bottom"
    ),
    margin=dict(t=100, b=50)
)

# Clean axes
fig.for_each_xaxis(lambda axis: axis.update(title=None))
fig.update_yaxes(title_text="Score (%)", row=1, col=1)
fig.show()