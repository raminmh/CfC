import json
import glob

# Load all tensorflow configurations
tf = {}
for fname in glob.glob('CFC/config/*.json'):
    name = fname.replace('CFC/config\\','').replace('.json','')

    with open(fname, 'r') as f:
        data = f.read()
        data = json.loads(data)
        tf[name] = data