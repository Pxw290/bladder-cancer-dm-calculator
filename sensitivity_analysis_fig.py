# ============================================================================
#  膀胱癌 DM:敏感性分析图(AUC across cohorts + 外部AUC森林)—— Nature 风格
#  pip install xgboost lightgbm scikit-learn matplotlib openpyxl
#  数据:E:/BCA2/1.xlsx (Sheet4) | GROUP:1训练 2内部 3外部 | M:0/1
# ============================================================================
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

DATA="E:/BCA2/1.xlsx"; OUT="E:/BCA2/"
FEAT=["AGE.GROUP","PRIMARY.SITE","HISTOLOGICAL.TYPE","GRADE","T","N","TUMOR.SIZE.GROUP","NUMBER.OF.TUMORS"]
NPG_blue,NPG_green,NPG_red,NPG_purple="#3C5488","#00A087","#E64B35","#8491B4"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":7,"axes.titlesize":9,"axes.labelsize":8,"axes.linewidth":0.6,"axes.spines.top":False,
 "axes.spines.right":False,"legend.frameon":False,"savefig.dpi":600,"savefig.bbox":"tight"})

# ---- DeLong 单 AUC 方差 ----
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def auc_var(y,p):
    o=np.argsort(-y); yy=y[o]; pr=p[o]; m=int(yy.sum()); n=len(pr)-m
    tx=_midrank(pr[:m]); ty=_midrank(pr[m:]); tz=_midrank(pr)
    auc=tz[:m].sum()/m/n-(m+1)/2/n
    return float(auc), float(np.var((tz[:m]-tx)/n,ddof=1)/m+np.var(1-(tz[m:]-ty)/m,ddof=1)/n)
def ci(av): a,v=av; se=np.sqrt(v); return (a,max(a-1.96*se,0),min(a+1.96*se,1))

def enc_train_eval(df):
    X=df[FEAT].astype("category"); y=df["M"].astype(int).values; g=df["GROUP"].astype(int).values
    enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(X[g==1])
    Xs=StandardScaler().fit(enc.transform(X)[g==1]).transform(enc.transform(X))
    mdl=GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42).fit(Xs[g==1],y[g==1])
    return {nm:auc_var(y[g==gg].astype(float),mdl.predict_proba(Xs[g==gg])[:,1]) for gg,nm in [(1,"Training"),(2,"Internal"),(3,"External")]}

def impute_grade(df,seed):   # 对未知分级(GRADE==3)做一次后验抽样插补
    rng=np.random.default_rng(seed); d2=df.copy(); fe=[f for f in FEAT if f!="GRADE"]
    kn=d2["GRADE"]!=3; un=d2["GRADE"]==3
    Xk=pd.get_dummies(d2.loc[kn,fe].astype("category"))
    clf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1).fit(Xk,d2.loc[kn,"GRADE"])
    Xu=pd.get_dummies(d2.loc[un,fe].astype("category")).reindex(columns=Xk.columns,fill_value=0)
    pr=clf.predict_proba(Xu); cl=clf.classes_
    d2.loc[un,"GRADE"]=[rng.choice(cl,p=p) for p in pr]; return d2
def mice_scenario(df,m=10):  # 多重插补 + Rubin 合并
    res={c:[] for c in ["Training","Internal","External"]}
    for i in range(m):
        r=enc_train_eval(impute_grade(df,i))
        for c in res: res[c].append(r[c])
    out={}
    for c in res:
        a=np.array([x[0] for x in res[c]]); w=np.array([x[1] for x in res[c]])
        Q=a.mean(); Tv=w.mean()+(1+1/m)*a.var(ddof=1); se=np.sqrt(Tv)
        out[c]=(Q,max(Q-1.96*se,0),min(Q+1.96*se,1))
    return out

# ---------------- 读数据 + 5 个场景 ----------------
d=pd.read_excel(DATA,sheet_name="Sheet4")
d.columns=[("HISTOLOGICAL.TYPE" if "ICD-O-3" in c else c) for c in d.columns]
scen={}
scen["Primary analysis\n(Full cohort)"]={c:ci(v) for c,v in enc_train_eval(d).items()}
scen["Excluding\nGrade Unknown"]        ={c:ci(v) for c,v in enc_train_eval(d[d["GRADE"]!=3]).items()}
scen["Urothelial\ncarcinoma only"]      ={c:ci(v) for c,v in enc_train_eval(d[d["HISTOLOGICAL.TYPE"]==1]).items()}
scen["Excluding\nT4 stage"]             ={c:ci(v) for c,v in enc_train_eval(d[d["T"]!=4]).items()}
scen["MICE imputation\n(Grade)"]        =mice_scenario(d,m=10)

# ---------------- 画图 ----------------
names=list(scen); cohorts=["Training","Internal","External"]
col={"Training":NPG_blue,"Internal":NPG_green,"External":NPG_red}
fig,(axA,axB)=plt.subplots(2,1,figsize=(7.2,8),gridspec_kw={"height_ratios":[1,0.9]})

x=np.arange(len(names)); off={"Training":-0.22,"Internal":0,"External":0.22}
for c in cohorts:
    a=[scen[n][c][0] for n in names]; lo=[scen[n][c][0]-scen[n][c][1] for n in names]; hi=[scen[n][c][2]-scen[n][c][0] for n in names]
    axA.errorbar(x+off[c],a,yerr=[lo,hi],fmt="o",ms=4,color=col[c],capsize=2,lw=.9,label=c,zorder=3)
    for xi,ai in zip(x+off[c],a): axA.text(xi,ai+0.012,f"{ai:.3f}",ha="center",va="bottom",fontsize=5.5,color=col[c])
axA.set_xticks(x); axA.set_xticklabels(names,fontsize=6.5)
axA.set_ylabel("Area under the curve (AUC)"); axA.set_ylim(0.78,0.905)
axA.set_title("Sensitivity analyses: AUC across three cohorts",fontweight="bold",loc="left",pad=16)
axA.annotate("Gradient Boosting Machine | error bars: 95% CI",xy=(0,1),xycoords="axes fraction",xytext=(0,3),textcoords="offset points",fontsize=6,color="grey",va="bottom")
axA.grid(axis="y",ls=":",lw=.4,alpha=.6); axA.legend(loc="lower right",ncol=3)

ext=sorted([(n,)+scen[n]["External"] for n in names],key=lambda r:r[1])
ref=scen["Primary analysis\n(Full cohort)"]["External"][0]
axB.axvline(ref,ls="--",lw=.8,color=NPG_red,alpha=.7)
for i,(n,a,lo,hi) in enumerate(ext):
    mice="MICE" in n; mk="^" if mice else "D"; cc=NPG_purple if mice else NPG_blue
    axB.errorbar(a,i,xerr=[[a-lo],[hi-a]],fmt=mk,ms=6 if mice else 5,color=cc,capsize=2.5,lw=1,zorder=3)
    axB.text(hi+0.004,i,f"{a:.3f} ({lo:.3f}-{hi:.3f})",va="center",fontsize=6)
axB.set_yticks(range(len(ext))); axB.set_yticklabels([r[0].replace(chr(10)," ") for r in ext],fontsize=6.5)
axB.set_xlabel("Area under the curve (AUC)"); axB.set_xlim(0.80,0.90)
axB.set_title("Sensitivity analyses: external validation AUC",fontweight="bold",loc="left",pad=16)
axB.annotate("Gradient Boosting Machine | 95% confidence interval",xy=(0,1),xycoords="axes fraction",xytext=(0,3),textcoords="offset points",fontsize=6,color="grey",va="bottom")
axB.legend(handles=[Line2D([0],[0],marker="D",color="w",markerfacecolor=NPG_blue,ms=6,label="Sensitivity analysis"),
                    Line2D([0],[0],marker="^",color="w",markerfacecolor=NPG_purple,ms=7,label="MICE imputation")],loc="upper left",fontsize=6)
fig.tight_layout(h_pad=2.2)
fig.savefig(OUT+"Fig_Sensitivity_AUC.pdf"); fig.savefig(OUT+"Fig_Sensitivity_AUC.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
print("done ->",OUT)
