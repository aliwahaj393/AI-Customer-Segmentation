import json

def fix_notebook():
    with open('Project.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                # For df
                if "df[col].fillna(df[col].mode()[0], inplace=True)" in line:
                    line = line.replace("df[col].fillna(df[col].mode()[0], inplace=True)", "df[col] = df[col].fillna(df[col].mode()[0])")
                elif "df[col].fillna(df[col].median(), inplace=True)" in line:
                    line = line.replace("df[col].fillna(df[col].median(), inplace=True)", "df[col] = df[col].fillna(df[col].median())")
                
                # For test_df_clean
                elif "test_df_clean[col].fillna(test_df_clean[col].mode()[0], inplace=True)" in line:
                    line = line.replace("test_df_clean[col].fillna(test_df_clean[col].mode()[0], inplace=True)", "test_df_clean[col] = test_df_clean[col].fillna(test_df_clean[col].mode()[0])")
                elif "test_df_clean[col].fillna(test_df_clean[col].median(), inplace=True)" in line:
                    line = line.replace("test_df_clean[col].fillna(test_df_clean[col].median(), inplace=True)", "test_df_clean[col] = test_df_clean[col].fillna(test_df_clean[col].median())")
                
                new_source.append(line)
            cell['source'] = new_source

    with open('Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Inplace fillna fixed successfully.")

if __name__ == '__main__':
    fix_notebook()
