import json, numpy as np, pandas as pd
from scipy import stats
from k2_replication import SUBSET_MAP, MANIFEST, RESULTS

ZEROSHOT=["TimesFM-2.5","FlowState-9.1M","Kairos_10m","Kairos_23m","Kairos_50m","sundial_base_128m",
 "YingLong_110m","YingLong_300m","YingLong_50m","YingLong_6m","granite-flowstate-r1","TiRex",
 "DeOSAlphaTimeGPTPredictor-2025","Toto_Open_Base_1.0","TabPFN-TS","Moirai_base","Moirai_large",
 "Moirai_small","VisionTS"]
PRETRAINED=["Moirai2","chronos_bolt_base","chronos_bolt_small","Chronos_large","Chronos_base",
 "Chronos_small","TimesFM","timesfm_2_0_500m","TTM-R1-Pretrained","TTM-R2-Pretrained","Lag-Llama"]
DEEP=["xLSTM-Mixer","PatchTST","TFT","N-BEATS","DLinear","TIDE","DeepAR","Crossformer","iTransformer"]
STAT=["Auto_Arima","Seasonal_Naive","Auto_Theta","Auto_ETS","Naive"]
GROUPS={"zero-shot":ZEROSHOT,"pretrained":PRETRAINED,"deep":DEEP,"statistical":STAT}

r=pd.read_csv(RESULTS); r["base"]=r.dataset.str.split("/").str[0]
nv=r.groupby("dataset")["num_variates"].first()
MV=sorted({d.split("/")[0] for d in nv[nv>1].index})

raw=json.load(open(MANIFEST)); a=pd.DataFrame(raw["active_dev_series"])
a=a[a.subset.isin(SUBSET_MAP)].copy(); a["base"]=a.subset.map(lambda s:SUBSET_MAP[s][0])

def omega(col="omega_levels", drop_mv=True, agg="mean"):
    d=a[~a.base.isin(MV)] if drop_mv else a
    return d.groupby("base")[col].agg(agg).rename("omega").reset_index()

def table(metric_col, om, agg="mean"):
    f={k:r[r.model.isin(v)][["base",metric_col]].dropna().groupby("base")[metric_col].agg(agg)
       for k,v in GROUPS.items()}
    return pd.DataFrame(f).reset_index().merge(om,on="base").dropna()

def rho(tab,A,B):
    d=100*(tab[A]-tab[B])/tab[A]; ok=np.isfinite(d)
    return stats.spearmanr(tab.omega[ok],d[ok])

METRICS={"sMAPE (K2a, paper's metric)":"eval_metrics/sMAPE[0.5]",
         "MSE (K2a, paper's check)":"eval_metrics/MSE[mean]",
         "MAE = 2x pinball@0.5 (K2b)":"eval_metrics/MAE[0.5]",
         "CRPS (bonus comparator)":"eval_metrics/mean_weighted_sum_quantile_loss"}

print("="*78); print("K2a / K2b under the paper's exact taxonomy"); print("="*78)
print(f"{'metric':<30}{'zs vs deep':>22}{'zs vs statistical':>24}")
print("-"*78)
om=omega()
for name,col in METRICS.items():
    t=table(col,om)
    r1,p1=rho(t,"zero-shot","deep"); r2,p2=rho(t,"zero-shot","statistical")
    print(f"{name:<30}{r1:>+12.3f} (p={p1:.3f}){r2:>+14.3f} (p={p2:.3f})")
print(f"\npaper: zs vs deep -0.556 | zs vs statistical -0.234   (n={len(om)} here vs 28 in paper)")

print("\n"+"="*78); print("ROBUSTNESS: does the zero-shot vs deep result survive my choices?"); print("="*78)
M="eval_metrics/sMAPE[0.5]"
variants=[("mean within family, Omega levels",     omega(), "mean"),
          ("BEST within family (min sMAPE)",       omega(), "min"),
          ("MEDIAN within family",                 omega(), "median"),
          ("Omega on first DIFFERENCES",           omega("omega_diffs"), "mean"),
          ("Omega aggregated by MEDIAN not mean",  omega(agg="median"), "mean"),
          ("INCLUDE flattened multivariate",       omega(drop_mv=False), "mean")]
print(f"{'variant':<40}{'n':>4}{'zs vs deep':>16}{'zs vs stat':>16}")
print("-"*78)
for label,om_v,agg in variants:
    t=table(M,om_v,agg)
    r1,p1=rho(t,"zero-shot","deep"); r2,p2=rho(t,"zero-shot","statistical")
    print(f"{label:<40}{len(t):>4}{r1:>+10.3f} p={p1:.2f}{r2:>+10.3f} p={p2:.2f}")
