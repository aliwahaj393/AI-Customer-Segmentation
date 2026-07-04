import json

def fix_notebook():
    with open('Project.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                # Fix select_dtypes
                if "include='object'" in line:
                    line = line.replace("include='object'", "exclude='number'")
                if 'include="object"' in line:
                    line = line.replace('include="object"', "exclude='number'")
                
                # Fix sparse=False for scikit-learn 1.4+
                if "sparse=False" in line:
                    line = line.replace("sparse=False", "sparse_output=False")
                
                new_source.append(line)
            cell['source'] = new_source

    with open('Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("OHE and select_dtypes fixed successfully.")

if __name__ == '__main__':
    fix_notebook()
