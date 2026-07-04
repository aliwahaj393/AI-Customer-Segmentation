import json

def update_notebook():
    with open('Project.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Modify the DecisionTreeClassifier
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'clf = DecisionTreeClassifier(random_state=42)' in source:
                new_source = source.replace(
                    'clf = DecisionTreeClassifier(random_state=42)',
                    'clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)'
                )
                
                # Split back into lines with newlines included
                lines = new_source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
                
    # 2. Add a new cell to plot the tree
    plot_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- 6. Visualize Decision Tree ---\n",
            "from sklearn.tree import plot_tree\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "plt.figure(figsize=(20,10))\n",
            "plot_tree(clf, feature_names=X_train.columns, class_names=[str(c) for c in clf.classes_], filled=True, rounded=True, fontsize=10)\n",
            "plt.title('Decision Tree Rules for Customer Segments')\n",
            "plt.show()\n"
        ]
    }
    
    # Find where to insert (after the confusion matrix cell)
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Confusion Matrix' in source and 'sns.heatmap' in source:
                insert_idx = i + 1
                break
                
    if insert_idx != -1:
        nb['cells'].insert(insert_idx, plot_cell)
        print("Successfully added plot cell.")
    else:
        print("Could not find confusion matrix cell, appending plot cell to end.")
        nb['cells'].append(plot_cell)

    with open('Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook updated successfully.")

if __name__ == '__main__':
    update_notebook()
