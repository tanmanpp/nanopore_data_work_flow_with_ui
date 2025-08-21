# main.py
import os
import re
import io
import json
import time
import asyncio
import zipfile
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')
RAW_MAP_PY = os.path.join(TOOLS_DIR, 'raw_read_to_mapping_genome.py')

# ============================
# Path utilities
# ============================

def is_windows_path(p: str) -> bool:
    return bool(re.match(r'^[A-Za-z]:\\', p)) or p.startswith('\\\\')
    
def win_to_wsl(p: str) -> str:
    r"""Best effort convert Windows absolute path to WSL-like POSIX path.
    Examples:
      D:\\data\\x -> /mnt/d/data/x
      \\\\server\\share\\x (UNC) -> /mnt/unc/server/share/x
      If already POSIX, return unchanged.
    """
    if not p:
        return p
    if p.startswith('/'):
        return p
    if is_windows_path(p):
        # UNC path
        if p.startswith('\\\\'):
            rest = p.lstrip('\\')
            rest = rest.replace('\\', '/')
            return '/mnt/unc/' + rest
        # Drive letter
        m = re.match(r'^([A-Za-z]):\\(.*)$', p)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).replace('\\', '/')
            return f'/mnt/{drive}/{rest}'
    # Fallback: replace backslashes
    return p.replace('\\', '/')

def norm_abs(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = p.strip()
    if not p:
        return p
    return win_to_wsl(p)

def ensure_dir(path_dir: str):
    if not path_dir or path_dir.startswith('<'):
        return
    os.makedirs(path_dir, exist_ok=True)

def basename_no_ext(path: str) -> str:
    name = os.path.basename(path.rstrip('/'))
    for suf in ('.fastq.gz', '.fq.gz', '.fastq', '.fq', '.fasta.gz', '.fasta', '.fa'):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name

# ============================
# Job system (SSE logs)
# ============================

def _now_ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')

@dataclass
class Job:
    job_id: str
    status: str = 'queued'   # queued | running | succeeded | failed
    return_code: Optional[int] = None
    log_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

JOBS: Dict[str, Job] = {}

async def _pipe_reader(stream: asyncio.StreamReader, job: Job, prefix: str):
    while True:
        line = await stream.readline()
        if not line:
            break
        try:
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            await job.log_queue.put(f'[{_now_ts()}] [{prefix}] {text}')
        except Exception:
            pass

async def _run_shell_with_logs(job: Job, cmd: str, env: Optional[Dict[str,str]] = None):
    start = time.time()
    await job.log_queue.put(f'[{_now_ts()}] [job] {job.job_id} START')
    await job.log_queue.put(f'[{_now_ts()}] [cmd] {cmd}')

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    # Concurrently read stdout/stderr
    readers = []
    if proc.stdout is not None:
        readers.append(asyncio.create_task(_pipe_reader(proc.stdout, job, 'stdout')))
    if proc.stderr is not None:
        readers.append(asyncio.create_task(_pipe_reader(proc.stderr, job, 'stderr')))

    # Wait readers & process
    try:
        if readers:
            await asyncio.gather(*readers)
    finally:
        rc = await proc.wait()
    job.return_code = rc
    elapsed = time.time() - start
    await job.log_queue.put(f'[{_now_ts()}] [job] {job.job_id} FINISH rc={rc} elapsed={elapsed:.2f}s')
    return rc

async def run_job_shell(cmd: str) -> Job:
    job = Job(job_id=str(int(time.time()*1000)))
    JOBS[job.job_id] = job
    job.status = 'running'
    async def worker():
        try:
            rc = await _run_shell_with_logs(job, cmd)
            job.status = 'succeeded' if rc == 0 else 'failed'
        except Exception as e:
            await job.log_queue.put(f'[{_now_ts()}] [error] {type(e).__name__}: {e}')
            job.status = 'failed'
            job.return_code = -1
        finally:
            await job.log_queue.put('[DONE]')
            job.done_event.set()
    asyncio.create_task(worker())
    return job

async def run_job_pipeline(cmds: List[str], post_py: Optional[Callable[[Job], Awaitable[None]]] = None) -> Job:
    job = Job(job_id=str(int(time.time()*1000)))
    JOBS[job.job_id] = job
    job.status = 'running'

    async def worker():
        try:
            total = len(cmds)
            for idx, c in enumerate(cmds, start=1):
                step_start = time.time()
                await job.log_queue.put(f'[{_now_ts()}] [pipeline] step {idx}/{total} START')
                await job.log_queue.put(f'[{_now_ts()}] [cmd] {c}')
                rc = await _run_shell_with_logs(job, c)
                step_elapsed = time.time() - step_start
                await job.log_queue.put(f'[{_now_ts()}] [pipeline] step {idx}/{total} END rc={rc} elapsed={step_elapsed:.2f}s')
                if rc != 0:
                    job.status = 'failed'
                    job.return_code = rc
                    await job.log_queue.put(f'[{_now_ts()}] [pipeline] stop due to rc={rc}')
                    break
            else:
                if post_py:
                    try:
                        await post_py(job)
                    except Exception as e:
                        await job.log_queue.put(f'[{_now_ts()}] [post][error] {type(e).__name__}: {e}')
                        job.status = 'failed'
                        job.return_code = -1
                        return
                job.status = 'succeeded'
                job.return_code = 0
        except Exception as e:
            await job.log_queue.put(f'[{_now_ts()}] [error] {type(e).__name__}: {e}')
            job.status = 'failed'
            job.return_code = -1
        finally:
            await job.log_queue.put('[DONE]')
            job.done_event.set()
    asyncio.create_task(worker())
    return job

# ============================
# FastAPI app
# ============================

app = FastAPI(title='NGS Mapping UI Backend', version='2025-08-12')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount("/ui", StaticFiles(directory=Path(__file__).with_name("ui"), html=True), name="ui")

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/ui/v1.3_20250812_zh.html")

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/ui/v1.3_20250812_en.html")


# ============================
# Models
# ============================

class WSPrepareReq(BaseModel):
    ws_abs: str

class BasecallReq(BaseModel):
    data_dir: str
    out_dir: str
    device: str = 'cuda:all'
    min_qscore: int = 9
    kit_name: str
    model_tier: str

class DemuxReq(BaseModel):
    in_fastq_list: List[str]
    out_dir: str
    kit_name: Optional[str] = None
    threads: int = 10

class PorechopReq(BaseModel):
    in_fastq: str
    out_dir: str
    threads: int = 10

class NanoPlotReq(BaseModel):
    in_fastq: str
    out_dir: str
    threads: int = 10

class KrakenReq(BaseModel):
    in_fastq: str
    db: str
    out_path: str
    threads: int = 10

class RecentrifugeReq(BaseModel):
    kraken_result: str
    out_html: str
    nodes_file: str

class RefFetchReq(BaseModel):
    dest_dir: str
    local_fasta: Optional[str] = None
    ncbi_id: Optional[str] = None
    build_index: bool = True

class MappingReq(BaseModel):
    f1: str
    VG: str
    t: int = 10
    sid: str
    oD: str

# ============================
# Helpers for command quoting
# ============================

def q(s: str) -> str:
    return shlex.quote(s)

def build_basecall_cmd(req: BasecallReq) -> str:
    data_dir = norm_abs(req.data_dir)
    out_dir = norm_abs(req.out_dir)
    ensure_dir(out_dir)
    parts = [
        'dorado basecaller',
        f'--device {req.device}',
        f'--min-qscore {int(req.min_qscore)}',
        f'--kit-name {q(req.kit_name)}',
        '--emit-fastq',
        f'-o {q(out_dir)}',
        '--no-trim',
        req.model_tier,
        q(data_dir)
    ]
    return ' '.join(parts)

def build_demux_cmd(req: DemuxReq) -> str:
    out_dir_root = norm_abs(req.out_dir)
    out_dir = os.path.join(out_dir_root, 'demux')
    ensure_dir(out_dir)
    kit = f'--kit-name {q(req.kit_name)}' if req.kit_name else ''
    inputs = ' '.join(q(norm_abs(x)) for x in req.in_fastq_list)
    parts = [
        'dorado demux',
        f'--output-dir {q(out_dir)}',
        kit,
        f'-t {int(req.threads)}',
        '--emit-fastq',
        inputs
    ]
    return ' '.join([p for p in parts if p])

def build_porechop_cmd(req: PorechopReq) -> str:
    in_abs = norm_abs(req.in_fastq)
    out_root = norm_abs(req.out_dir)
    bn = basename_no_ext(in_abs)
    m = BARCODE_RE.search(bn)
    out_dir = os.path.join(out_root, m.group(1).lower()) if m else out_root
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f'{bn}_trimmed.fastq')
    parts = ['porechop', f'-i {q(in_abs)}', f'-o {q(out_path)}', f'-t {int(req.threads)}']
    return ' '.join(parts)

def build_nanoplot_cmd(req: NanoPlotReq) -> str:
    in_abs = norm_abs(req.in_fastq)
    out_root = norm_abs(req.out_dir)
    bn = basename_no_ext(in_abs)
    m = BARCODE_RE.search(bn)
    out_dir = os.path.join(out_root, m.group(1).lower()) if m else out_root
    ensure_dir(out_dir)
    parts = ['NanoPlot', f'-t {int(req.threads)}', f'-o {q(os.path.join(out_dir, f"nanoplot_{bn}"))}', f'--fastq {q(in_abs)}', '--only-report']
    return ' '.join(parts)

def build_kraken_cmd(req: KrakenReq) -> str:
    in_abs = norm_abs(req.in_fastq)
    db = norm_abs(req.db)
    out_full = norm_abs(req.out_path)
    out_dir, out_name = os.path.dirname(out_full), os.path.basename(out_full)
    m = BARCODE_RE.search(basename_no_ext(in_abs))
    if m:
        out_dir = os.path.join(out_dir, m.group(1).lower())
    ensure_dir(out_dir)
    out_full = os.path.join(out_dir, out_name)
    parts = ['k2', 'classify', f'--threads {int(req.threads)}', f'--output {q(out_full)}',f'--db {q(db)}', q(in_abs)]
    return ' '.join(parts)

def build_rcf_cmd(req: RecentrifugeReq) -> str:
    k = norm_abs(req.kraken_result); o = norm_abs(req.out_html)
    in_dir, in_name = os.path.dirname(k), os.path.basename(k)
    out_dir, out_name = os.path.dirname(o), os.path.basename(o)
    # 推回輸入 fastq 的 barcode（從 kraken 輸出檔名推不準，改從 in_name 推）
    m = BARCODE_RE.search(basename_no_ext(in_name))
    if m:
        in_dir  = os.path.join(in_dir,  m.group(1).lower())
        out_dir = os.path.join(out_dir, m.group(1).lower())
    ensure_dir(out_dir)
    k_out = os.path.join(in_dir, in_name)
    o_out = os.path.join(out_dir, out_name)
    cmd = [
        'conda', 'run', '-n', 'recentrifuge_env',
        'rcf',
        f'-n {q(norm_abs(req.nodes_file))}',
        f'-k {q(k_out)}',
        f'-o {q(o_out)}',
    ]
    return ' '.join(cmd)

def build_fetch_ref_cmd(req: RefFetchReq) -> List[str]:
    dest = norm_abs(req.dest_dir)
    ensure_dir(dest)
    cmds: List[str] = []
    if req.ncbi_id:
        fasta = os.path.join(dest, f'ref_{req.ncbi_id}.fasta')
        cmds.append(f'efetch -db nuccore -id {q(req.ncbi_id)} -format fasta > {q(fasta)}')
        if req.build_index:
            cmds.append(f'minimap2 -d {q(fasta)}.mmi {q(fasta)}')
    elif req.local_fasta:
        # Only index local fasta into dest
        src = norm_abs(req.local_fasta)
        base = basename_no_ext(src)
        mmi = os.path.join(dest, f'{base}.mmi')
        cmds.append(f'minimap2 -d {q(mmi)} {q(src)}')
    else:
        raise HTTPException(400, 'Provide ncbi_id or local_fasta')
    return cmds

def build_mapping_cmd(req: MappingReq) -> str:
    in_abs = norm_abs(req.f1)
    vg = norm_abs(req.VG)
    out_root = norm_abs(req.oD)
    # sid 優先，其次從檔名抓
    sid = re.sub(r'\s+', '_', req.sid.strip()) if req.sid else basename_no_ext(in_abs)
    out_dir = os.path.join(out_root, sid)
    ensure_dir(out_dir)
    parts = [
        f'python {q(RAW_MAP_PY)}',
        f'-f1 {q(in_abs)}',
        '-mt minimap2',
        f'-VG {q(vg)}',
        f'-t {int(req.t)}',
        f'-sid {q(sid)}',
        f'-oD {q(out_dir)}'
    ]
    return ' '.join(parts)

# ============================
# Post-process helpers
# ============================

BARCODE_RE = re.compile(r'(barcode\d{2})', re.IGNORECASE)

async def post_demux_reorganize(job: Job, out_dir: str):
    out_dir = norm_abs(out_dir)
    await job.log_queue.put(f'[{_now_ts()}] [post] reorganizing demux outputs in {out_dir}')
    try:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if not fn.lower().endswith(('.fastq', '.fastq.gz', '.fq', '.fq.gz')):
                    continue
                m = BARCODE_RE.search(fn)
                if not m:
                    continue
                barcode = m.group(1).lower()
                src = os.path.join(root, fn)
                dst_dir = os.path.join(out_dir, barcode)
                ensure_dir(dst_dir)
                # target filename
                ext = '.fastq.gz' if fn.lower().endswith('.fastq.gz') or fn.lower().endswith('.fq.gz') else '.fastq'
                dst = os.path.join(dst_dir, f'{barcode}{ext}')
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(src, dst)
                    await job.log_queue.put(f'[{_now_ts()}] [post] {fn} -> {barcode}/{os.path.basename(dst)}')
                except Exception as e:
                    await job.log_queue.put(f'[{_now_ts()}] [post][warn] move failed: {fn}: {e}')
    except Exception as e:
        await job.log_queue.put(f'[{_now_ts()}] [post][error] {e}')

# ============================
# API endpoints
# ============================

@app.post('/api/workspace/prepare')
async def ws_prepare(req: WSPrepareReq):
    ws = norm_abs(req.ws_abs)
    if not ws or ws.startswith('<'):
        raise HTTPException(400, 'Invalid workspace path')
    steps = [
        'Step_1_basecaller_demux',
        'Step_2_Trimming_QC',
        'Step_3_Kraken_result',
        'Step_4_mapping_consensus',
    ]
    for s in steps:
        ensure_dir(os.path.join(ws, s))
    return {'ok': True, 'workspace': ws}

@app.post('/api/dorado/basecall')
async def dorado_basecall(req: BasecallReq):
    cmd = build_basecall_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/dorado/demux')
async def dorado_demux(req: DemuxReq):
    cmd = build_demux_cmd(req)
    out_dir = norm_abs(req.out_dir)
    async def post_py(job: Job):
        await post_demux_reorganize(job, out_dir)
    job = await run_job_pipeline([cmd], post_py=post_py)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/porechop/run')
async def porechop_run(req: PorechopReq):
    cmd = build_porechop_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/nanoplot/run')
async def nanoplot_run(req: NanoPlotReq):
    cmd = build_nanoplot_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/kraken2/run')
async def kraken2_run(req: KrakenReq):
    cmd = build_kraken_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/recentrifuge/run')
async def recentrifuge_run(req: RecentrifugeReq):
    cmd = build_rcf_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

@app.post('/api/reference/fetch')
async def reference_fetch(req: RefFetchReq):
    cmds = build_fetch_ref_cmd(req)
    # We also want to return the fasta path to front-end (for mapVG)
    fasta_out = None
    if req.ncbi_id:
        fasta_out = os.path.join(norm_abs(req.dest_dir), f'ref_{req.ncbi_id}.fasta')
    elif req.local_fasta:
        # return the local_fasta as the reference for mapping
        fasta_out = norm_abs(req.local_fasta)
    async def post_py(job: Job):
        await job.log_queue.put(f'[{_now_ts()}] [post] reference fetch/index finished')
    job = await run_job_pipeline(cmds, post_py=post_py)
    return {'job_id': job.job_id, 'status': job.status, 'fasta': fasta_out}

@app.post('/api/mapping/run')
async def mapping_run(req: MappingReq):
    cmd = build_mapping_cmd(req)
    job = await run_job_shell(cmd)
    return {'job_id': job.job_id, 'status': job.status}

# ---------- Jobs status & events (SSE) ----------

@app.get('/api/jobs/{job_id}')
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'job not found')
    return {'job_id': job.job_id, 'status': job.status, 'return_code': job.return_code}

@app.get('/api/jobs/{job_id}/events')
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'job not found')

    async def event_gen():
        # flush any queued logs
        while True:
            try:
                item = await asyncio.wait_for(job.log_queue.get(), timeout=1.0)
                yield f'data: {item}\n\n'
                if item == '[DONE]':
                    break
            except asyncio.TimeoutError:
                if job.done_event.is_set():
                    # drain remaining
                    while not job.log_queue.empty():
                        item2 = await job.log_queue.get()
                        yield f'data: {item2}\n\n'
                    break
                else:
                    # keep-alive comment line (avoids proxies closing idle connections)
                    yield ': keep-alive\n\n'

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
    }
    return StreamingResponse(event_gen(), headers=headers)

# ---------- File utilities ----------

@app.get('/api/file')
async def get_file(path: str = Query(..., description='absolute path')):
    p = norm_abs(path)
    if not os.path.exists(p):
        raise HTTPException(404, 'file not found')
    # naive content-type
    media = 'text/html' if p.lower().endswith('.html') else 'application/octet-stream'
    return FileResponse(p, media_type=media)

@app.get('/api/files/zip')
async def zip_dir(path: str = Query(...)):
    p = norm_abs(path)
    if not os.path.exists(p):
        raise HTTPException(404, 'path not found')
    # create in-memory zip
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(p):
            base = os.path.basename(p.rstrip('/'))
            for root, _, files in os.walk(p):
                for fn in files:
                    full = os.path.join(root, fn)
                    arc = os.path.join(base, os.path.relpath(full, p))
                    zf.write(full, arc)
        else:
            zf.write(p, os.path.basename(p))
    mem.seek(0)
    headers = {'Content-Disposition': f'attachment; filename={os.path.basename(p.rstrip("/"))}.zip'}
    return Response(content=mem.read(), headers=headers, media_type='application/zip')

# ---------- FASTQ compare ----------

def _iter_fastq(fp):
    while True:
        h = fp.readline()
        if not h:
            return
        s = fp.readline()
        p = fp.readline()
        q = fp.readline()
        if not q:
            return
        yield s.rstrip().decode('utf-8', errors='ignore'), q.rstrip().decode('utf-8', errors='ignore')

def compute_fastq_stats(path: str, max_reads: Optional[int] = None):
    import gzip
    path = norm_abs(path)
    opener = gzip.open if path.endswith('.gz') else open
    n = 0
    total_len = 0
    lengths = []
    qsum = 0
    bases = 0
    with opener(path, 'rb') as fp:
        for seq, qual in _iter_fastq(fp):
            l = len(seq)
            total_len += l
            lengths.append(l)
            bases += l
            # mean q: sum per-base
            qsum += sum(max(0, (ord(c) - 33)) for c in qual)
            n += 1
            if max_reads and n >= max_reads:
                break
    if n == 0:
        return {'reads': 0, 'mean_len': 0, 'n50': 0, 'mean_q': 0.0}
    lengths.sort()
    # N50
    half = total_len / 2
    acc = 0
    n50 = 0
    for L in reversed(lengths):
        acc += L
        if acc >= half:
            n50 = L
            break
    mean_q = qsum / bases if bases > 0 else 0.0
    return {'reads': n, 'mean_len': round(total_len / n, 2), 'n50': int(n50), 'mean_q': round(mean_q, 2)}

@app.post('/api/fastq/compare')
async def fastq_compare(payload: Dict[str, str]):
    pre = payload.get('pre_fastq')
    post = payload.get('post_fastq')
    if not pre or not post:
        raise HTTPException(400, 'pre_fastq and post_fastq are required')
    # compute (limit reads for speed? choose None = full; you can adjust below)
    pre_stats = compute_fastq_stats(pre, max_reads=None)
    post_stats = compute_fastq_stats(post, max_reads=None)
    # CSV
    csv = 'dataset,reads,mean_len,n50,mean_q\n'
    csv += f'pre,{pre_stats["reads"]},{pre_stats["mean_len"]},{pre_stats["n50"]},{pre_stats["mean_q"]}\n'
    csv += f'post,{post_stats["reads"]},{post_stats["mean_len"]},{post_stats["n50"]},{post_stats["mean_q"]}\n'
    return Response(content=csv, media_type='text/csv')
