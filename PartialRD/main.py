import pandas as pd
from partialRD.PartialRD.source.schemadesign import SchemaDesign
from partialRD.PartialRD.source.partialRD import PartialRD
import time
if __name__ == '__main__':
    file_path= '../dataset/Indusdata_running_example.csv'
    r_df = pd.read_csv(file_path)
    r_df= r_df.drop_duplicates()
    r = r_df.to_dict(orient="records")
    n=len(r_df)
    afd_list = [
    {"lhs": ["timestamp"], "rhs": ["click"]},
    {"lhs": ["uin"], "rhs": ["age", "sex", "device"]},
    {"lhs": ["docid"], "rhs": ["subcategory"]},
    {"lhs": ["subcategory"], "rhs": ["category"]},
    ]
    t1= time.time()
    relation_schemas = SchemaDesign(afd_list, r_df.columns)
    decomposed_relations = PartialRD(r,r_df,afd_list,relation_schemas,n)
    print(time.time()-t1)
    for name, rows in decomposed_relations.items():
        df = pd.DataFrame(rows)
        df.to_csv(f"./output/{name}.csv", index=False)
        print(f"{name} has {len(rows)} rows and {len(df.columns)} columns.")
