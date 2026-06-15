# ============================================================================
#  膀胱癌 DM 预测:Fig 4–7  (Nature 风格)
#  运行前:pip install xgboost lightgbm shap scikit-learn matplotlib seaborn openpyxl
#  数据:E:/BCA2/1.xlsx  (Sheet4)  |  GROUP:1训练 2内部 3外部  |  M:0/1 结局
# ============================================================================
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, shap
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------- 路径 / 变量 ----------------
DATA = "E:/BCA2/1.xlsx"; OUT = "E:/BCA2/"
FEATURES = ["AGE.GROUP","PRIMARY.SITE","HISTOLOGICAL.TYPE","GRADE","T","N","TUMOR.SIZE.GROUP","NUMBER.OF.TUMORS"]
NICE     = ["Age group","Primary site","Histology","Grade","T stage","N stage","Tumor size","Tumor number"]

# ---------------- Nature 风格 ----------------
NPG=["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148"]
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":7,"axes.titlesize":9,"axes.labelsize":8,"axes.linewidth":0.6,"axes.edgecolor":"black",
 "axes.spines.top":False,"axes.spines.right":False,"xtick.major.width":0.6,"ytick.major.width":0.6,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.2,"legend.frameon":False,
 "figure.dpi":120,"savefig.dpi":600,"savefig.bbox":"tight"})

# ---------------- 读数据 + 编码 ----------------
d = pd.read_excel(DATA, sheet_name="Sheet4")
d.columns = [("HISTOLOGICAL.TYPE" if "ICD-O-3" in c else c) for c in d.columns]  # 重命名组织学列
y = d["M"].astype(int).values; grp = d["GROUP"].astype(int).values
Xcat = d[FEATURES].astype("category")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xcat[grp==1])
fn = list(enc.get_feature_names_out(FEATURES))
Xoh = enc.transform(Xcat)
Xs  = StandardScaler().fit(Xoh[grp==1]).transform(Xoh)          # one-hot+标准化(给 LR/SVM/KNN 等)
sp  = {n:(Xs[grp==g], y[grp==g]) for g,n in [(1,"Training"),(2,"Internal"),(3,"External")]}

# ---------------- 9 个模型 ----------------
def get_models():
    return {"LR":LogisticRegression(max_iter=2000,class_weight="balanced"),
     "RF":RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1),
     "XGBoost":XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,scale_pos_weight=14,eval_metric="logloss",random_state=42,n_jobs=-1),
     "LightGBM":LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1),
     "GBDT":GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42),
     "AdaBoost":AdaBoostClassifier(n_estimators=300,learning_rate=0.5,random_state=42),
     "SVM":SVC(probability=True,class_weight="balanced",random_state=42),
     "KNN":KNeighborsClassifier(n_neighbors=25),"NaiveBayes":GaussianNB()}

# ===================== Fig 4:9模型交叉验证 ROC =====================
def fig4():
    Xtr,ytr=sp["Training"]; cv=StratifiedKFold(5,shuffle=True,random_state=42); mf=np.linspace(0,1,200)
    fig,ax=plt.subplots(figsize=(3.5,3.5))
    for i,(name,mdl) in enumerate(get_models().items()):
        tprs,aucs=[],[]
        for ti,vi in cv.split(Xtr,ytr):
            m=clone(mdl).fit(Xtr[ti],ytr[ti]); p=m.predict_proba(Xtr[vi])[:,1]
            fpr,tpr,_=roc_curve(ytr[vi],p); t=np.interp(mf,fpr,tpr); t[0]=0; tprs.append(t); aucs.append(auc(fpr,tpr))
        mt=np.mean(tprs,0); mt[-1]=1
        ax.plot(mf,mt,color=NPG[i],lw=1.3,label=f"{name} ({np.mean(aucs):.3f}\u00b1{np.std(aucs):.3f})")
    ax.plot([0,1],[0,1],"--",lw=.7,color="grey")
    ax.set_xlabel("1 \u2212 Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Five-fold cross-validation ROC",fontweight="bold",loc="left")
    ax.legend(loc="lower right",title="Model (mean AUC\u00b1SD)",title_fontsize=6.2)
    fig.savefig(OUT+"Fig4_CV_ROC.pdf"); fig.savefig(OUT+"Fig4_CV_ROC.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); plt.close()

# ===================== DeLong 检验 =====================
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong_p(y_true,p1,p2):
    o=np.argsort(-y_true); yy=y_true[o]; pr=np.vstack((p1[o],p2[o])); m=int(yy.sum()); n=pr.shape[1]-m; k=2
    tx=np.array([_midrank(pr[r,:m]) for r in range(k)]); ty=np.array([_midrank(pr[r,m:]) for r in range(k)])
    tz=np.array([_midrank(pr[r,:]) for r in range(k)])
    aucs=tz[:,:m].sum(1)/m/n-(m+1)/2/n; v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n; l=np.array([[1,-1]]); z=(aucs[0]-aucs[1])/np.sqrt(l@cov@l.T)[0,0]
    return 2*stats.norm.sf(abs(z))

# ===================== Fig 5:AUC 热图 + DeLong =====================
def fig5():
    models=get_models(); coh=["Training","Internal","External"]; proba={}; tab=np.zeros((len(models),3))
    for i,(name,mdl) in enumerate(models.items()):
        m=clone(mdl).fit(*sp["Training"]); proba[name]={}
        for j,c in enumerate(coh):
            Xc,yc=sp[c]; p=m.predict_proba(Xc)[:,1]; proba[name][c]=(yc,p); tab[i,j]=roc_auc_score(yc,p)
    names=list(models); order=np.argsort(-tab[:,2]); no=[names[k] for k in order]
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.6))
    sns.heatmap(tab[order],annot=True,fmt=".3f",cmap="YlGnBu",vmin=.65,vmax=.85,xticklabels=coh,
                yticklabels=no,ax=ax[0],cbar_kws={"label":"AUC","shrink":.7},annot_kws={"size":6.5},linewidths=.5)
    ax[0].set_title("a  AUC across cohorts",fontweight="bold",loc="left"); ax[0].tick_params(length=0)
    K=len(no); P=np.ones((K,K))
    for a in range(K):
        for b in range(a+1,K):
            ya,pa=proba[no[a]]["External"]; _,pb=proba[no[b]]["External"]
            P[a,b]=P[b,a]=delong_p(ya.astype(float),pa,pb)
    sns.heatmap(P,mask=np.eye(K,dtype=bool),annot=True,fmt=".2f",cmap="Reds_r",vmin=0,vmax=.1,xticklabels=no,
                yticklabels=no,ax=ax[1],cbar_kws={"label":"DeLong P","shrink":.7},annot_kws={"size":5.5},linewidths=.5)
    ax[1].set_title("b  DeLong test (External)",fontweight="bold",loc="left"); ax[1].tick_params(length=0,labelsize=6)
    ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=45,ha="right")
    fig.tight_layout(); fig.savefig(OUT+"Fig5_AUC_DeLong.pdf"); fig.savefig(OUT+"Fig5_AUC_DeLong.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); plt.close()

# ===================== Fig 6:校准 + DCA =====================
def fig6(model_name="GBDT"):
    best=clone(get_models()[model_name]).fit(*sp["Training"]); coh=["Training","Internal","External"]; cols=[NPG[3],NPG[2],NPG[0]]
    pro={c:(sp[c][1],best.predict_proba(sp[c][0])[:,1]) for c in coh}
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.5))
    ax[0].plot([0,1],[0,1],"--",lw=.7,color="grey",label="Ideal")
    for c,col in zip(coh,cols):
        yv,pv=pro[c]; fr,mp=calibration_curve(yv,pv,n_bins=10,strategy="quantile")
        ax[0].plot(mp,fr,"o-",color=col,ms=3,lw=1.2,label=f"{c} (Brier={brier_score_loss(yv,pv):.3f})")
    ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("a  Calibration",fontweight="bold",loc="left"); ax[0].set_xlim(0,.6); ax[0].set_ylim(0,.6); ax[0].legend(loc="upper left")
    def nb(yv,pv,th):
        out=[]; n=len(yv)
        for t in th:
            pr=pv>=t; tp=np.sum(pr&(yv==1)); fp=np.sum(pr&(yv==0)); out.append(tp/n-(fp/n)*(t/(1-t)))
        return np.array(out)
    th=np.linspace(.01,.6,100)
    for c,col in zip(coh,cols):
        yv,pv=pro[c]; ax[1].plot(th,nb(yv,pv,th),color=col,lw=1.3,label=c)
    yv,_=pro["External"]; prev=yv.mean()
    ax[1].plot(th,prev-(1-prev)*(th/(1-th)),color="grey",lw=.9,label="Treat all")
    ax[1].plot(th,np.zeros_like(th),"--",color="black",lw=.9,label="Treat none")
    ax[1].set_xlabel("Threshold probability"); ax[1].set_ylabel("Net benefit")
    ax[1].set_title("b  Decision curve analysis",fontweight="bold",loc="left"); ax[1].set_xlim(0,.6); ax[1].set_ylim(-.02,prev*1.15); ax[1].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT+"Fig6_Calibration_DCA.pdf"); fig.savefig(OUT+"Fig6_Calibration_DCA.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); plt.close()

# ===================== Fig 7:SHAP =====================
def fig7():
    X=d[FEATURES].astype(int).copy(); X.columns=NICE; tr=grp==1
    mdl=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=14,eval_metric="logloss",random_state=42,n_jobs=-1).fit(X[tr],y[tr])
    SV=shap.TreeExplainer(mdl)(X[tr]); sv=SV.values; imp=np.abs(sv).mean(0); order=np.argsort(imp); top=NICE[int(np.argmax(imp))]
    fig=plt.figure(figsize=(7.6,6.8))
    ax1=fig.add_subplot(2,2,1); ax1.barh([NICE[i] for i in order],[imp[i] for i in order],color=NPG[3],height=.65)
    ax1.set_xlabel("Mean |SHAP value|"); ax1.set_title("a  Feature importance",fontweight="bold",loc="left")
    ax2=fig.add_subplot(2,2,2); plt.sca(ax2); shap.summary_plot(sv,X[tr],max_display=8,show=False,plot_size=None)
    ax2.set_title("b  SHAP summary (beeswarm)",fontweight="bold",loc="left")
    ax3=fig.add_subplot(2,2,3); plt.sca(ax3); shap.dependence_plot(top,sv,X[tr],interaction_index=None,show=False,ax=ax3,color=NPG[0])
    ax3.set_title(f"c  Dependence: {top}",fontweight="bold",loc="left")
    idx=int(np.argmax(mdl.predict_proba(X[tr])[:,1])); ax4=fig.add_subplot(2,2,4); plt.sca(ax4)
    shap.plots.waterfall(SV[idx],max_display=9,show=False); ax4.set_title("d  Waterfall (one high-risk case)",fontweight="bold",loc="left")
    fig.tight_layout(); fig.savefig(OUT+"Fig7_SHAP.pdf"); fig.savefig(OUT+"Fig7_SHAP.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); plt.close()

if __name__=="__main__":
    fig4(); print("Fig4 OK")
    fig5(); print("Fig5 OK")
    fig6(); print("Fig6 OK")     # 默认用 GBDT(外部AUC最高);换模型: fig6("LightGBM")
    fig7(); print("Fig7 OK")
    print("全部完成 → E:/BCA2/")
