import json, numpy as np, pandas as pd
from scipy import stats
from k2_replication import SUBSET_MAP, RESULTS

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

def load(path, drop_dup_jena=True):
    a=pd.DataFrame(json.load(open(path))["active_dev_series"])
    a=a[a.subset.isin(SUBSET_MAP)].copy()
    if drop_dup_jena: a=a[a.subset!="jena_weather"]   # duplicate of jena_weather_10T
    a["base"]=a.subset.map(lambda s:SUBSET_MAP[s][0]); return a

def omega(a, col="omega_levels", agg="mean"):
    return a.groupby("base")[col].agg(agg).rename("omega").reset_index()

def tab(om, metric, agg="mean"):
    f={k:r[r.model.isin(v)][["base",metric]].dropna().groupby("base")[metric].agg(agg) for k,v in GROUPS.items()}
    return pd.DataFrame(f).reset_index().merge(om,on="base").dropna()

def rho(t,A,B):
    d=100*(t[A]-t[B])/t[A]; ok=np.isfinite(d)
    return stats.spearmanr(t.omega[ok],d[ok])

NEW="newman/Dev Manifest.json"; OLD="/mnt/user-data/uploads/Dev_Manifest_from_Algoverse_AI_Research.json"
an=load(NEW); ao=load(OLD)
M="eval_metrics/sMAPE[0.5]"
print("base datasets:  OLD(flattened, no trim) %d   NEW(channel-split, trimmed) %d"
      % (ao.base.nunique(), an.base.nunique()))
print("NEW bases:", sorted(an.base.unique()))
print()
print("="*76); print("K2a / K2b on the CORRECTED manifest"); print("="*76)
METRICS={"sMAPE (paper's metric)":"eval_metrics/sMAPE[0.5]","MSE":"eval_metrics/MSE[mean]",
 "MAE = 2x pinball@0.5 (K2b)":"eval_metrics/MAE[0.5]","CRPS":"eval_metrics/mean_weighted_sum_quantile_loss"}
om=omega(an)
print(f"{'metric':<30}{'n':>4}{'zs vs deep':>20}{'zs vs stat':>20}")
print("-"*76)
for name,col in METRICS.items():
    t=tab(om,col); r1,p1=rho(t,"zero-shot","deep"); r2,p2=rho(t,"zero-shot","statistical")
    print(f"{name:<30}{len(t):>4}{r1:>+12.3f} p={p1:.3f}{r2:>+12.3f} p={p2:.3f}")
print(f"\npaper: zs vs deep -0.556 | zs vs stat -0.234 | n=28")

print("\n"+"="*76); print("ROBUSTNESS on corrected data"); print("="*76)
print(f"{'variant':<42}{'n':>4}{'zs vs deep':>16}{'zs vs stat':>16}")
print("-"*76)
for lbl,o_,agg in [("mean within family",omega(an),"mean"),
                   ("BEST within family",omega(an),"min"),
                   ("MEDIAN within family",omega(an),"median"),
                   ("Omega on first differences",omega(an,"omega_diffs"),"mean"),
                   ("Omega aggregated by median",omega(an,agg="median"),"mean")]:
    t=tab(o_,M,agg); r1,p1=rho(t,"zero-shot","deep"); r2,p2=rho(t,"zero-shot","statistical")
    print(f"{lbl:<42}{len(t):>4}{r1:>+10.3f} p={p1:.2f}{r2:>+10.3f} p={p2:.2f}")

om.to_csv("omega_by_dataset_corrected.csv",index=False)
tab(om,M).round(3).to_csv("k2a_corrected_table.csv",index=False)
print("\nPer-dataset (sMAPE):"); print(tab(om,M).sort_values("omega").round(3).to_string(index=False))
