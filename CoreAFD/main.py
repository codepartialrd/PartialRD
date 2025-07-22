import pandas as pd
import pickle as pkl
from partialRD.CoreAFD.source.quickcore import QuickCore
import time
from partialRD.CoreAFD.source.selectingcoreafd import cache_and_index



def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=' ', sep=',', header=0)
def afds_discovered_by_sota(AFD_PKL):
    with open(AFD_PKL, 'rb') as f:
        sigma, error_dict = pkl.load(f)
    return sigma, error_dict

if __name__=="__main__":
    CSV_PATH= "../dataset/Indusdata_running_example.csv"
    AFD_PKL = "afds_discovered_by_dafd/afds.pkl"
    sigma, error = afds_discovered_by_sota(AFD_PKL)
    df = read_csv(CSV_PATH)
    t0=time.time()
    qc = QuickCore(df, sigma, error)
    S_quickcore = qc.quickcore()
    elapsed = time.time() - t0
    print(f"[QuickCore] Time of filtering and pruning: {elapsed:.4f} seconds")
    print(f"[QuickCore] Number of promising AFD sets: {len(S_quickcore)}")
    t3=time.time()
    coreafd, dg=cache_and_index(S_quickcore, df, df.columns)
    print(f"Core AFD set: {coreafd}")
    print(f"[Core AFD] Time of selecting core afd set from S: {time.time() - t3:.4f} seconds")




