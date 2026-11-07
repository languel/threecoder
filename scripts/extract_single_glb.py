import base64
import json
import os
import re

content = open('/Users/liuboto/.gemini/antigravity/brain/b2fca7ef-a39a-4fa6-8d86-b31719ecc24d/.system_generated/steps/961/output.txt').read()
start = content.find('{')
end = content.rfind('}') + 1
data = json.loads(content[start:end])
b64 = data['dataUrl'].split(',', 1)[1]
raw = base64.b64decode(b64)
out_dir = 'benchmarks/eval10/glbs/qwen35_4b_glb_bf16'
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'P01_default_cube.glb')
with open(out_file, 'wb') as f:
    f.write(raw)
print(f'Wrote {out_file}: {len(raw)} bytes, magic: {raw[:4]}')
