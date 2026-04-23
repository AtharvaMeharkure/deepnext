import json

file_path = 'Untitled1 (1).ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'lm3d = det.faces[0].landmarks_3d' in line and '[:468]' not in line:
                source[i] = line.replace('lm3d = det.faces[0].landmarks_3d', 'lm3d = det.faces[0].landmarks_3d[:468]  # FORCE exactly 468 points')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook fixed successfully!")
