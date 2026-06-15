# ============================================================================
#  Table 4:九模型性能汇总(AUC/Sens/Spec/F1/Brier + Accuracy)× 三队列
#  pip install xgboost lightgbm scikit-learn openpyxl
#  阈值=训练集 Youden 最优切点(固定应用到内部/外部)
# ============================================================================
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, brier_score_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA="E:/BCA2/1.xlsx"; OUT="E:/BCA2/"
FEAT=["AGE.GROUP","PRIMARY.SITE","HISTOLOGICAL.TYPE","GRADE","T","N","TUMOR.SIZE.GROUP","NUMBER.OF.TUMORS"]

d=pd.read_excel(DATA,sheet_name="Sheet4")
d.columns=[("HISTOLOGICAL.TYPE" if "ICD-O-3" in c else c) for c in d.columns]
y=d["M"].astype(int).values; g=d["GROUP"].astype(int).values
X=d[FEAT].astype("category")
enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(X[g==1])
Xs=StandardScaler().fit(enc.transform(X)[g==1]).transform(enc.transform(X))
sp={nm:(Xs[g==gg],y[g==gg]) for gg,nm in [(1,"Training"),(2,"Internal"),(3,"External")]}

models={"LR":LogisticRegression(max_iter=2000,class_weight="balanced"),
 "RF":RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1),
 "XGBoost":XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=14,eval_metric="logloss",random_state=42,n_jobs=-1),
 "LightGBM":LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1),
 "GBDT":GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42),
 "AdaBoost":AdaBoostClassifier(n_estimators=300,learning_rate=0.5,random_state=42),
 "SVM":SVC(probability=True,class_weight="balanced",random_state=42),
 "KNN":KNeighborsClassifier(n_neighbors=25),"NaiveBayes":GaussianNB()}

def _mr(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1; i=j
    o=np.empty(N); o[J]=T; return o
def auc_ci(yv,p):
    o=np.argsort(-yv); yy=yv[o]; pr=p[o]; m=int(yy.sum()); n=len(pr)-m
    tx=_mr(pr[:m]); ty=_mr(pr[m:]); tz=_mr(pr); a=tz[:m].sum()/m/n-(m+1)/2/n
    se=np.sqrt(np.var((tz[:m]-tx)/n,ddof=1)/m+np.var(1-(tz[m:]-ty)/m,ddof=1)/n)
    return a,max(a-1.96*se,0),min(a+1.96*se,1)

rows=[]
for name,mdl in models.items():
    m=clone(mdl).fit(*sp["Training"])
    ytr,ptr=sp["Training"][1],m.predict_proba(sp["Training"][0])[:,1]
    fpr,tpr,thr=roc_curve(ytr,ptr); thr_opt=thr[np.argmax(tpr-fpr)]
    for c in ["Training","Internal","External"]:
        Xc,yc=sp[c]; p=m.predict_proba(Xc)[:,1]; pred=(p>=thr_opt).astype(int)
        a,lo,hi=auc_ci(yc.astype(float),p); tn,fp,fn,tp=confusion_matrix(yc,pred).ravel()
        rows.append({"Model":name,"Cohort":c,"AUC":f"{a:.3f} ({lo:.3f}-{hi:.3f})",
            "Accuracy":f"{accuracy_score(yc,pred):.3f}","Sensitivity":f"{tp/(tp+fn):.3f}",
            "Specificity":f"{tn/(tn+fp):.3f}","F1":f"{f1_score(yc,pred):.3f}","Brier":f"{brier_score_loss(yc,p):.3f}"})
tab=pd.DataFrame(rows)
order=(tab[tab.Cohort=="External"].assign(a=lambda x:x.AUC.str[:5].astype(float)).sort_values("a",ascending=False)["Model"].tolist())
tab["Model"]=pd.Categorical(tab["Model"],categories=order,ordered=True)
tab=tab.sort_values(["Model","Cohort"]).reset_index(drop=True)
tab.to_csv(OUT+"Table4_model_performance.csv",index=False)
tab.to_excel(OUT+"Table4_model_performance.xlsx",index=False)
print(tab.to_string(index=False))
