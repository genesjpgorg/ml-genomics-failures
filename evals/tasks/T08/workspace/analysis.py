import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

x = pd.read_csv("expression.csv").set_index("patient")
y = pd.read_csv("response.csv").set_index("patient").loc[x.index, "responder"].values

# pick the 20 genes most associated with response
selector = SelectKBest(f_classif, k=20).fit(x.values, y)
signature = x.columns[selector.get_support()]

# evaluate the signature by 5-fold cross-validation
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(LogisticRegression(max_iter=1000), x[signature].values, y, cv=cv, method="predict_proba")[:, 1]
print("cv AUROC", round(roc_auc_score(y, p), 3))
