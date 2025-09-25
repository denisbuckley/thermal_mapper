
#!/usr/bin/env python3
"""
batch_run_thermals_v1.py  (proximity-based + GeoJSON)
...
"""
import argparse, json, math, subprocess
from pathlib import Path
from typing import List, Dict
import pandas as pd, numpy as np

CIRCLES_SCRIPT = "circles_from_brecords_v1d.py"
CIRCLE_CLUSTERS_SCRIPT = "circle_clusters_v1s.py"
ALT_CLUSTERS_SCRIPT = "overlay_altitude_clusters_v1c.py"
MATCH_SCRIPT = "match_clusters_v1.py"

def run(cmd: List[str], verbose: bool) -> None:
    if verbose: print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0: raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")

def ensure_dir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def clean_existing(paths: List[Path], verbose: bool) -> None:
    for p in paths:
        if p.exists():
            try: p.unlink(); 
            except Exception as e: print(f"[WARN] Could not remove {p}: {e}")
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R=6371000.0; import math as m
    phi1=m.radians(float(lat1)); phi2=m.radians(float(lat2))
    dphi=m.radians(float(lat2)-float(lat1)); dlmb=m.radians(float(lon2)-float(lon1))
    a=m.sin(dphi/2)**2+m.cos(phi1)*m.cos(phi2)*m.sin(dlmb/2)**2; c=2*m.atan2(m.sqrt(a),m.sqrt(1-a))
    return R*c
class UnionFind:
    def __init__(self,n:int): self.parent=list(range(n)); self.rank=[0]*n
    def find(self,x:int)->int:
        while self.parent[x]!=x: self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a:int,b:int)->None:
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return
        if self.rank[ra]<self.rank[rb]: self.parent[ra]=rb
        elif self.rank[ra]>self.rank[rb]: self.parent[rb]=ra
        else: self.parent[rb]=ra; self.rank[ra]+=1
def cluster_by_radius(latitudes: np.ndarray, longitudes: np.ndarray, eps_m: float) -> np.ndarray:
    n=len(latitudes); uf=UnionFind(n)
    for i in range(n):
        for j in range(i+1,n):
            d=haversine_m(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            if d<=eps_m: uf.union(i,j)
    labels=np.array([uf.find(i) for i in range(n)],dtype=int)
    root_map={}; next_id=0
    for r in labels:
        if r not in root_map: root_map[r]=next_id; next_id+=1
    return np.array([root_map[r] for r in labels],dtype=int)
def aggregate_cluster(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return df.groupby(label_col).agg(
        n_points=("cluster_label","size"),
        lat=("lat","mean"), lon=("lon","mean"),
        max_climb_ms=("strength_ms","max"),
        median_climb_ms=("strength_ms","median"),
        median_alt_gain_m=("alt_gain_m","median"),
    ).reset_index(drop=True)
def to_geojson_points(df: pd.DataFrame) -> dict:
    feats=[]
    for _,row in df.iterrows():
        feats.append({
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[float(row["lon"]),float(row["lat"])]},
            "properties":{
                "n_points":int(row.get("n_points",1)),
                "max_climb_ms": float(row["max_climb_ms"]) if pd.notna(row.get("max_climb_ms", np.nan)) else None,
                "median_climb_ms": float(row["median_climb_ms"]) if pd.notna(row.get("median_climb_ms", np.nan)) else None,
                "median_alt_gain_m": float(row["median_alt_gain_m"]) if pd.notna(row.get("median_alt_gain_m", np.nan)) else None,
            }
        })
    return {"type":"FeatureCollection","features":feats}
def main():
    ap=argparse.ArgumentParser(description="Batch IGC runner (proximity-based) with GeoJSON export")
    ap.add_argument("--igc-dir",default="igc"); ap.add_argument("--outdir",default="outputs")
    ap.add_argument("--max-dist-m",type=float,default=5000.0)
    ap.add_argument("--min-climb-ms",type=float,default=1.0)
    ap.add_argument("--min-alt-gain-m",type=float,default=500.0)
    ap.add_argument("--eps-m",type=float,default=300.0); ap.add_argument("--min-samples",type=int,default=1)
    ap.add_argument("--geojson-out",default=None); ap.add_argument("--verbose",action="store_true")
    args=ap.parse_args()
    igc_dir=Path(args.igc_dir); outdir=Path(args.outdir); ensure_dir(outdir)
    igc_files=sorted(igc_dir.glob("*.igc"))
    if not igc_files: print(f"[INFO] No .igc files found in {igc_dir.resolve()}"); return
    all_rows=[]
    for igc in igc_files:
        stem=igc.stem; flight_outdir=outdir/stem; ensure_dir(flight_outdir)
        circle_out=flight_outdir/"circles.csv"; circle_clusters=flight_outdir/"circle_clusters_enriched.csv"
        alt_clusters=flight_outdir/"overlay_altitude_clusters.csv"; matched_out=flight_outdir/"matched_clusters.csv"
        for p in (circle_out,circle_clusters,alt_clusters,matched_out):
            if p.exists():
                try: p.unlink(); 
                except Exception as e: print(f"[WARN] Could not remove {p}: {e}")
        py=sys.executable
        run([py,CIRCLES_SCRIPT,str(igc)],args.verbose)
        run([py,CIRCLE_CLUSTERS_SCRIPT,str(igc)],args.verbose)
        run([py,ALT_CLUSTERS_SCRIPT,str(igc)],args.verbose)
        run([py,MATCH_SCRIPT,str(circle_clusters),str(alt_clusters),str(matched_out)],args.verbose)
        if not matched_out.exists(): print(f"[WARN] No matched output for {igc.name}; skipping"); continue
        m=pd.read_csv(matched_out); if m.empty: 
            print(f"[INFO] Matched output empty for {igc.name}") if args.verbose else None; continue
        m["strength_ms"]=m[["circle_av_climb_ms","alt_av_climb_ms"]].max(axis=1,skipna=True)
        m["alt_gain_m"]=m[["circle_alt_gained_m","alt_alt_gained_m"]].max(axis=1,skipna=True)
        lat=m["circle_lat"].fillna(m["alt_lat"]); lon=m["circle_lon"].fillna(m["alt_lon"])
        keep=(m["dist_m"].fillna(np.inf)<=args.max_dist_m)&(m["strength_ms"].fillna(0)>=args.min_climb_ms)&(m["alt_gain_m"].fillna(0)>=args.min_alt_gain_m)&lat.notna()&lon.notna()
        kept=m.loc[keep].copy(); 
        if kept.empty: 
            print(f"[INFO] No matches passed thresholds for {igc.name}") if args.verbose else None; continue
        kept["lat"]=lat.loc[keep].values; kept["lon"]=lon.loc[keep].values; kept["flight_id"]=stem
        all_rows.append(kept[["flight_id","lat","lon","strength_ms","alt_gain_m","circle_cluster_id","alt_cluster_id","dist_m","time_overlap_s","overlap_frac"]])
    if not all_rows:
        print("[INFO] No matches across flights after filtering.")
        (outdir/"thermals_all_raw.csv").write_text("",encoding="utf-8")
        (outdir/"thermals_clusters.csv").write_text("",encoding="utf-8")
        gjp=Path(args.geojson_out) if args.geojson_out else (outdir/"thermals_clusters.geojson")
        gjp.write_text('{"type":"FeatureCollection","features":[]}',encoding="utf-8"); return
    all_df=pd.concat(all_rows,ignore_index=True); all_out=outdir/"thermals_all_raw.csv"; all_df.to_csv(all_out,index=False); print(f"[OK] Wrote {all_out} ({len(all_df)} rows)")
    labels=cluster_by_radius(all_df["lat"].to_numpy(), all_df["lon"].to_numpy(), eps_m=args.eps_m); all_df["cluster_label"]=labels
    counts=all_df["cluster_label"].value_counts(); keep_labels=set(counts[counts>=args.min_samples].index.tolist()); clustered=all_df[all_df["cluster_label"].isin(keep_labels)].copy()
    agg=aggregate_cluster(clustered,"cluster_label"); clusters_out=outdir/"thermals_clusters.csv"; agg.to_csv(clusters_out,index=False); print(f"[OK] Wrote {clusters_out} ({len(agg)} clusters)")
    gjp=Path(args.geojson_out) if args.geojson_out else (outdir/"thermals_clusters.geojson")
    gj=to_geojson_points(agg); gjp.write_text(json.dumps(gj),encoding="utf-8"); print(f"[OK] Wrote {gjp}")
if __name__=="__main__": 
    import sys; main()
