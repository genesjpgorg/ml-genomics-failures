import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

s = pd.read_csv("samples.csv"); x = pd.read_csv("protein_matrix.csv").set_index("sample_id").loc[s.sample_id].values
y = (s.status == "case").astype(int).values
model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=12), LogisticRegression(max_iter=1000))
p = cross_val_predict(model, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("cv AUROC", round(roc_auc_score(y, p), 3))
