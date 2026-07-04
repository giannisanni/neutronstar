#!/usr/bin/env python3
"""Regenerate ds4_cuda_glm_stubs.inc: stubs for every ds4_gpu_* function
declared in ds4_gpu.h but not implemented in ds4_cuda.cu (excluding the
stubs file itself)."""
import re
hdr = open('ds4_gpu.h').read()
cu = open('ds4_cuda.cu').read()
cu = cu.split('#include "ds4_cuda_glm_stubs.inc"')[0]  # ignore stub include
import glob
for inc in glob.glob('ds4_cuda_glm_*.inc'):
    if 'stubs' in inc: continue
    cu += open(inc).read()
hdr_fns = re.findall(r'^(int|void|uint32_t|uint64_t|bool|float)\s+(ds4_gpu_[a-z0-9_]+)\s*\(', hdr, re.M)
stubs = []
for ret, name in hdr_fns:
    if re.search(r'extern "C"[^\n]*[ \*]' + name + r'\(', cu):
        continue
    m = re.search(r'^' + re.escape(ret) + r'\s+' + re.escape(name) + r'\s*\(([\s\S]*?)\);', hdr, re.M)
    if not m:
        continue
    args = m.group(1).strip()
    body = f'''extern "C" {ret} {name}({args}) {{
    static int warned;
    if (!warned) {{
        warned = 1;
        fprintf(stderr, "ds4: CUDA backend missing kernel (GLM port WIP): {name}\\n");
    }}'''
    for a in args.split(','):
        a = a.strip()
        if not a or a == 'void':
            continue
        an = re.sub(r'\[.*\]', '', a.split()[-1]).lstrip('*')
        body += f'\n    (void){an};'
    body += '\n    return 0;\n}' if ret != 'void' else '\n}'
    stubs.append(body)
open('ds4_cuda_glm_stubs.inc', 'w').write(
    "/* Auto-generated (tools/glm_stubgen.py). Stubs for backend functions the\n"
    " * CUDA GLM port has not implemented yet. */\n\n" + '\n\n'.join(stubs) + '\n')
print(f"{len(stubs)} stubs remaining")
