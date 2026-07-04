import json
import subprocess
import sys
import os

def verify_notebook():
    try:
        with open('Project.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print("Error loading notebook:", e)
        return

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code_cells.append(''.join(cell['source']))

    # Patch matplotlib and seaborn to not block
    patch = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
"""
    
    script_content = patch + '\n'.join(code_cells)
    
    with open('temp_run.py', 'w', encoding='utf-8') as f:
        f.write(script_content)

    print("Running notebook code...")
    result = subprocess.run([sys.executable, 'temp_run.py'], capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode == 0:
        print("SUCCESS: The notebook ran successfully with no errors!")
    else:
        print("FAILED: The notebook encountered an error during execution:")
        print(result.stderr)
        
    # Clean up
    if os.path.exists('temp_run.py'):
        os.remove('temp_run.py')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify_notebook()
