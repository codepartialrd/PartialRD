import pandas as pd
from partialRD.PartialRD.source.schemadesign import SchemaDesign
from partialRD.PartialRD.source.partialRD import PartialRD
if __name__ == '__main__':
    file_path= '../dataset/Indusdata_running_example.csv'
    r_df = pd.read_csv(file_path)
    r = r_df.to_dict(orient="records")
    afd_list = [
    {"lhs": ["timestamp"], "rhs": ["click"]},
    {"lhs": ["uin"], "rhs": ["age", "sex", "device"]},
    {"lhs": ["docid"], "rhs": ["subcategory"]},
    {"lhs": ["subcategory"], "rhs": ["category"]},
    ]
    relation_schemas = SchemaDesign(afd_list, r_df.columns)
    decomposed_relations = PartialRD(r,afd_list,relation_schemas)
    for name, rows in decomposed_relations.items():
        df = pd.DataFrame(rows)
        df.to_csv(f"./output/{name}.csv", index=False)