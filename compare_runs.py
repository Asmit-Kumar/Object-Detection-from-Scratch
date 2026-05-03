import json
from collections import defaultdict
import os

def load_run(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        data = json.load(f)
    if 'results' not in data:
        return None
    
    conf_by_digit = defaultdict(list)
    sources = set()
    for item in data['results']:
        conf_by_digit[item['digit']].append(item['confidence'])
        sources.add(item.get('bbox_source', 'unknown'))
        
    mean_conf = {}
    for d in sorted(conf_by_digit.keys()):
        mean_conf[d] = sum(conf_by_digit[d]) / len(conf_by_digit[d])
        
    return {
        "mean_conf": mean_conf,
        "sources": sources,
        "num_samples": len(data['results'])
    }

runs = {
    "run_01 (CNN + Classical fallback)": "logs/run_01.json",
    "run_03 (CNN Only - faster but less accurate box)": "logs/run_03.json",
    "run_04 (CNN + Refined Classical Crop)": "logs/run_04.json",
    "run_torch (PyTorch Fully Native - UltraLight)": "logs/run_torch.json",
}

results = {}
for name, path in runs.items():
    res = load_run(path)
    if res:
        results[name] = res

print("\n--- RESULTS COMPARISON ---\n")
for name, data in results.items():
    print(f"[{name}]")
    print(f"Samples: {data['num_samples']}, Sources: {data['sources']}")
    
    overall_mean = sum(data['mean_conf'].values()) / len(data['mean_conf'])
    print(f"Overall Mean Confidence: {overall_mean:.4f}")
    
    print("Per-digit confidence:")
    for d, c in data['mean_conf'].items():
        print(f"  Digit {d}: {c:.4f}")
    print()
