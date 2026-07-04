import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('train.csv')    
df = df.drop(columns=['ID', 'Segmentation'])
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(exclude='number').columns.tolist()
print("Cat cols:", cat_cols)
ohe = OneHotEncoder(sparse_output=False, drop='first')
enc = ohe.fit_transform(df[cat_cols])
enc_df = pd.DataFrame(enc,
                      columns=ohe.get_feature_names_out(cat_cols),
                      index=df.index)
proc_df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

print("proc_df missing values:")
print(proc_df.isnull().sum())

scaler = StandardScaler()
scaled = scaler.fit_transform(proc_df)

print("Number of NaNs in scaled:", np.isnan(scaled).sum())
print("Number of infs in scaled:", np.isinf(scaled).sum())

