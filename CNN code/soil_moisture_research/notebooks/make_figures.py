import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi":120,"savefig.dpi":200,"font.size":11,
    "axes.titlesize":12,"axes.grid":True,"grid.alpha":0.3,"axes.axisbelow":True})
FIGP = "figures"; RPT = "report/figures"
os.makedirs(FIGP,exist_ok=True); os.makedirs(RPT,exist_ok=True)
def save(fig,name):
    fig.tight_layout()
    fig.savefig(f"{FIGP}/{name}.png",bbox_inches="tight")
    fig.savefig(f"{RPT}/{name}.pdf",bbox_inches="tight")
    plt.close(fig); print("saved",name)

STAT_COL={"FD11":"#1f77b4","FD02":"#d62728","CB06":"#2ca02c","CB25":"#9467bd"}

# --- P1 lag correlation (Drive authoritative) ---
p1=pd.read_csv("results/P1_lag_correlations.csv")
fig,ax=plt.subplots(figsize=(7,4.5))
for s,g in p1.groupby("station"):
    g=g[g.lag_h>0]
    ax.plot(g.lag_h,g.corr_cumulative,marker="o",label=s,color=STAT_COL.get(s))
ax.set_xscale("log"); ax.set_xticks([1,2,3,6,12,24,48,72]); ax.set_xticklabels([1,2,3,6,12,24,48,72])
ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Corr(Ppt$_t$, $\\Delta$SWC$_{20}$ over lag)")
ax.set_title("Rainfall $\\rightarrow$ 20 cm soil-moisture response by lag"); ax.legend(title="Station")
save(fig,"fig_p1_lag_correlation")

# --- P4 depth comparison (5 vs 20 cm) ---
dc=pd.read_csv("results/P4_lag_depth_comparison.csv")
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,s in zip(axes.ravel(),["FD11","FD02","CB06","CB25"]):
    g=dc[(dc.station==s)&(dc.lag_h>0)]
    ax.plot(g.lag_h,g.corr_cumulative_5,marker="o",label="5 cm",color="#ff7f0e")
    ax.plot(g.lag_h,g.corr_cumulative_20,marker="s",label="20 cm",color="#1f77b4")
    ax.set_xscale("log"); ax.set_xticks([1,2,3,6,12,24,48,72]); ax.set_xticklabels([1,2,3,6,12,24,48,72])
    ax.set_title(s); ax.legend()
for ax in axes[-1]: ax.set_xlabel("Lag (hours)")
for ax in axes[:,0]: ax.set_ylabel("Cumulative corr.")
fig.suptitle("Rainfall correlation with soil moisture: 5 cm vs 20 cm depth",y=1.00)
save(fig,"fig_p4_lag_depth")

# --- RMSE vs horizon (SWC-only CNN vs persistence) ---
res=pd.read_csv("results/nov17_results.csv")
cnnA=res[res.config=="A: SWC_20 only"].groupby("horizon").RMSE.mean()
per=res[res.config=="persistence"].groupby("horizon").RMSE.mean()
fig,ax=plt.subplots(figsize=(7,4.5))
ax.plot(cnnA.index,cnnA.values,marker="o",label="SWC-only CNN",color="#1f77b4")
ax.plot(per.index,per.values,marker="s",ls="--",label="Persistence",color="#7f7f7f")
ax.set_xticks([24,48,72,96,168]); ax.set_xlabel("Forecast horizon (hours)")
ax.set_ylabel("Mean test RMSE (SWC$_{20}$)")
ax.set_title("Forecast error vs horizon (mean over 5 held-out station-years)"); ax.legend()
save(fig,"fig_rmse_vs_horizon")

# --- Feature importance heatmap: config x horizon ---
piv=pd.read_csv("results/feature_importance_by_horizon.csv",index_col=0)
order=["B: + existing rain","C: + lagged rain","D: + cumulative rain","E: + air temperature","F: combined met"]
piv=piv.loc[order]
fig,ax=plt.subplots(figsize=(7.5,4))
vmax=np.nanmax(np.abs(piv.values))
im=ax.imshow(piv.values,cmap="RdBu",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c}h" for c in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]
        ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=9,
                color="white" if abs(v)>vmax*0.6 else "black")
ax.set_title("%RMSE change vs SWC-only (blue=better, red=worse)")
fig.colorbar(im,ax=ax,label="% RMSE improvement")
save(fig,"fig_feature_importance_heatmap")

# --- Feature x station heatmap ---
cnn=res[res.architecture=="LargerCNN"]
pivs=cnn[cnn.config.isin(order)].pivot_table(index="config",columns="test_station",
      values="impr_vs_swc_%",aggfunc="mean").loc[order]
fig,ax=plt.subplots(figsize=(6.5,4))
vmax=np.nanmax(np.abs(pivs.values))
im=ax.imshow(pivs.values,cmap="RdBu",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(pivs.columns))); ax.set_xticklabels(pivs.columns)
ax.set_yticks(range(len(pivs.index))); ax.set_yticklabels(pivs.index)
for i in range(pivs.shape[0]):
    for j in range(pivs.shape[1]):
        v=pivs.values[i,j]
        ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=9,
                color="white" if abs(v)>vmax*0.6 else "black")
ax.set_title("%RMSE change vs SWC-only, by station")
fig.colorbar(im,ax=ax,label="% RMSE improvement")
save(fig,"fig_feature_by_station")

# --- Loss curves (representative) ---
lh=pd.read_csv("results/nov17_loss_histories.csv")
panels=[("A: SWC_20 only","FD11",2023,48),("A: SWC_20 only","CB25",2020,168),
        ("D: + cumulative rain","FD02",2020,48),("E: + air temperature","FD11",2020,72)]
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,(cfg,st,yr,h) in zip(axes.ravel(),panels):
    g=lh[(lh.config==cfg)&(lh.test_station==st)&(lh.test_year==yr)&(lh.horizon==h)]
    ax.plot(g.epoch,g.train_loss,label="train",color="#1f77b4")
    ax.plot(g.epoch,g.val_loss,label="val",color="#d62728")
    if len(g):
        be=g.loc[g.val_loss.idxmin(),"epoch"]
        ax.axvline(be,ls=":",color="gray",label=f"best ep {int(be)}")
    ax.set_title(f"{cfg.split(':')[0]} | {st}-{yr} H{h}"); ax.legend(fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
fig.suptitle("Training vs validation loss (representative runs)",y=1.00)
save(fig,"fig_loss_curves")
print("ALL FIGS DONE")
