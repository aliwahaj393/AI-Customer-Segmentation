import json

def fix_notebook():
    with open('Project.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                if "if df[col].dtype == 'object':" in line:
                    new_source.append(line.replace("if df[col].dtype == 'object':", "if not pd.api.types.is_numeric_dtype(df[col]):"))
                elif "if test_df_clean[col].dtype == 'object':" in line:
                    new_source.append(line.replace("if test_df_clean[col].dtype == 'object':", "if not pd.api.types.is_numeric_dtype(test_df_clean[col]):"))
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open('Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook fixed successfully.")

if __name__ == '__main__':
    fix_notebook()
