"""Machine B (macOS + iCloud): copy data/snakeclef to ~/snakeguard-data/snakeclef, re-downloading files iCloud
offloaded (16 in parallel). Resumable: finished files are skipped. Run from the project root:
    .venv/bin/python scripts/copy_data_out_of_icloud.py
"""
import os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
src = Path('data/snakeclef').resolve(); dst = Path.home() / 'snakeguard-data' / 'snakeclef'
files = [p for p in src.rglob('*') if p.is_file()]
def copy(p):
    q = dst / p.relative_to(src)
    size = p.stat().st_size
    if q.exists() and q.stat().st_size == size:
        return True
    q.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            data = p.read_bytes()
            if len(data) == size:
                tmp = q.with_name(q.name + '.part'); tmp.write_bytes(data); tmp.rename(q); return True
        except OSError:
            pass
        time.sleep(2)
    return False
local = [p for p in files if os.stat(p).st_blocks > 0]; evicted = [p for p in files if os.stat(p).st_blocks == 0]
print(len(files), 'files:', len(local), 'local,', len(evicted), 'evicted', flush=True)
t = time.time(); ok = [copy(p) for p in local]; print('local copied in %.0f s, failed %d' % (time.time() - t, ok.count(False)), flush=True)
t = time.time(); fail = []
with ThreadPoolExecutor(16) as ex:
    for k, (p, r) in enumerate(zip(evicted, ex.map(copy, evicted))):
        if not r: fail.append(str(p.relative_to(src)))
        if k % 500 == 0: print('evicted %d/%d, %.0f s' % (k, len(evicted), time.time() - t), flush=True)
print('done; failed:', len(fail), fail[:5], flush=True)
