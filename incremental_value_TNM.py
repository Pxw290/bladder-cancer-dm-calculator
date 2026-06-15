# ============================================================================
#  增量价值分析:全模型 vs 仅TNM(T+N)  —— ΔAUC(DeLong)+ IDI + 连续NRI
#  pip install scikit-learn matplotlib openpyxl scipy
#  数据:E:/BCA2/1.xlsx | GROUP:1训练 2内部 3外部 | M:0/1(结局)
# ============================================================================
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

DATA="E:/BCA2/1.xlsx"; OUT="E:/BCA2/"
FULL=["AGE.GROUP","PRIMARY.SITE","HISTOLOGICAL.TYPE","GRADE","T","N","TUMOR.SIZE.GROUP","NUMBER.OF.TUMORS"]
TNM=["T","N"]                                  # M 是结局,故 TNM 对照=T+N(临床分期)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":7,"axes.titlesize":9,"axes.labelsize":8,"axes.linewidth":0.6,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"savefig.dpi":600,"savefig.bbox":"tight"})

d=pd.read_excel(DATA,sheet_name="Sheet4")
d.columns=[("HISTOLOGICAL.TYPE" if "ICD-O-3" in c else c) for c in d.columns]
y=d["M"].astype(int).values; g=d["GROUP"].astype(int).values

def fit_predict(feats):
    X=d[feats].astype("category")
    enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(X[g==1])
    Xs=StandardScaler().fit(enc.transform(X)[g==1]).transform(enc.transform(X))
    mdl=GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42).fit(Xs[g==1],y[g==1])
    return mdl.predict_proba(Xs)[:,1]
p_full=fit_predict(FULL); p_tnm=fit_predict(TNM)
ext=g==3; ye=y[ext].astype(float); pf=p_full[ext]; pt=p_tnm[ext]

# ---- 指标 ----
def idi(yv,po,pn): ev=yv==1; ne=yv==0; return (pn[ev].mean()-pn[ne].mean())-(po[ev].mean()-po[ne].mean())
def nri_cont(yv,po,pn):
    ev=yv==1; ne=yv==0
    return (np.mean(pn[ev]>po[ev])-np.mean(pn[ev]<po[ev]))+(np.mean(pn[ne]<po[ne])-np.mean(pn[ne]>po[ne]))
def _mr(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1; i=j
    o=np.empty(N); o[J]=T; return o
def delong(yv,p1,p2):
    o=np.argsort(-yv); yy=yv[o]; pr=np.vstack((p1[o],p2[o])); m=int(yy.sum()); n=pr.shape[1]-m
    tx=np.array([_mr(pr[r,:m]) for r in range(2)]); ty=np.array([_mr(pr[r,m:]) for r in range(2)]); tz=np.array([_mr(pr[r,:]) for r in range(2)])
    auc=tz[:,:m].sum(1)/m/n-(m+1)/2/n; cov=np.cov((tz[:,:m]-tx)/n)/m+np.cov(1-(tz[:,m:]-ty)/m)/n
    z=(auc[0]-auc[1])/np.sqrt(np.array([[1,-1]])@cov@np.array([[1,-1]]).T)[0,0]
    return auc[0],auc[1],2*stats.norm.sf(abs(z))

af,at,pval=delong(ye,pf,pt); IDI=idi(ye,pt,pf); NRI=nri_cont(ye,pt,pf)
rng=np.random.default_rng(42); B=2000; bi=[];bn=[];ba=[]; idx=np.arange(len(ye))
for _ in range(B):
    sx=rng.choice(idx,len(idx),replace=True)
    if ye[sx].sum()<5: continue
    bi.append(idi(ye[sx],pt[sx],pf[sx])); bn.append(nri_cont(ye[sx],pt[sx],pf[sx]))
    ba.append(roc_auc_score(ye[sx],pf[sx])-roc_auc_score(ye[sx],pt[sx]))
ci=lambda v:(np.percentile(v,2.5),np.percentile(v,97.5))
dauc_ci,IDI_ci,NRI_ci=ci(ba),ci(bi),ci(bn)
print(f"AUC full={af:.3f} TNM={at:.3f} | ΔAUC={af-at:.3f} {dauc_ci} DeLong P={pval:.2e}")
print(f"IDI={IDI:.4f} {IDI_ci} | NRI(cont)={NRI:.4f} {NRI_ci}")

# ---- 图 ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={"width_ratios":[1,0.85]})
for p,lab,c in [(pf,f"Full model (AUC {af:.3f})","#E64B35"),(pt,f"TNM only (AUC {at:.3f})","#3C5488")]:
    fpr,tpr,_=roc_curve(ye,p); ax1.plot(fpr,tpr,color=c,lw=1.4,label=lab)
ax1.plot([0,1],[0,1],"--",lw=.7,color="grey"); ax1.set_xlabel("1 \u2212 Specificity"); ax1.set_ylabel("Sensitivity")
ax1.set_title("a  External validation ROC",fontweight="bold",loc="left")
ax1.legend(loc="lower right",title=f"DeLong P = {pval:.1e}",title_fontsize=6)
mets=[("\u0394AUC",af-at,dauc_ci),("IDI",IDI,IDI_ci),("NRI (cont.)",NRI,NRI_ci)]; yv=np.arange(3)[::-1]
ax2.axvline(0,ls="--",lw=.7,color="grey")
for (nm,est,(lo,hi)),yy in zip(mets,yv):
    ax2.errorbar(est,yy,xerr=[[est-lo],[hi-est]],fmt="o",ms=5,color="#00A087",capsize=3,lw=1)
    ax2.text(hi+0.02,yy,f"{est:.3f}\n({lo:.3f}, {hi:.3f})",va="center",fontsize=5.8)
ax2.set_yticks(yv); ax2.set_yticklabels([m[0] for m in mets]); ax2.set_xlim(-0.05,0.55)
ax2.set_xlabel("Improvement (Full vs TNM-only)"); ax2.set_title("b  Incremental value",fontweight="bold",loc="left")
fig.tight_layout(); fig.savefig(OUT+"Fig_Incremental_TNM.pdf")
fig.savefig(OUT+"Fig_Incremental_TNM.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
print("done ->",OUT)
