"""Extract the latest detection visualization from detection.ipynb outputs."""
import json, base64, sys, os
sys.stdout.reconfigure(encoding='utf-8')

NB = "notebooks/detection.ipynb"
OUT = "result/detection_output_latest.png"

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    for output in cell.get('outputs', []):
        # look for image/png outputs
        data = output.get('data', {})
        if 'image/png' in data:
            img_b64 = data['image/png']
            if isinstance(img_b64, list):
                img_b64 = ''.join(img_b64)
            img_bytes = base64.b64decode(img_b64)
            # only save if it's a reasonably large image (actual detection viz)
            if len(img_bytes) > 50000:
                with open(OUT, 'wb') as f_out:
                    f_out.write(img_bytes)
                print(f"Saved {OUT} ({len(img_bytes):,} bytes)")
                found = True
                break
    if found:
        break

if not found:
    print("No large image output found in notebook")
    # List all outputs found
    for i, cell in enumerate(nb['cells']):
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            if 'image/png' in data:
                img_b64 = data['image/png']
                if isinstance(img_b64, list):
                    img_b64 = ''.join(img_b64)
                print(f"  Cell {i}: image/png {len(base64.b64decode(img_b64)):,} bytes")
