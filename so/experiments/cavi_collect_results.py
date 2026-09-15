"""Read-only artifact collector; no model evaluation, no inferred pass values."""
from __future__ import annotations
import argparse, datetime, hashlib, io, json, subprocess, zipfile
from pathlib import Path

RUNS=(33961782977,33963051896,33963272140,33963491759)

def gh(path):
    return json.loads(subprocess.check_output(['gh','api',path],text=True))

def collect(out):
    out.mkdir(parents=True,exist_ok=False)
    index={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'breakthrough':False,'runs':[]}
    for run_id in RUNS:
        root=f'repos/xrey167/FinX-BB/actions/runs/{run_id}'
        run=gh(root)
        entry={k:run[k] for k in ('id','name','head_sha','status','conclusion','run_attempt')}
        entry['artifacts']=[]
        for a in gh(root+'/artifacts?per_page=100')['artifacts']:
            ar={k:a.get(k) for k in ('id','name','size_in_bytes','digest','expired')}
            entry['artifacts'].append(ar)
            if a['expired'] or a['size_in_bytes']>30_000_000:
                ar['collection']='SKIPPED_EXPIRED_OR_SIZE';continue
            folder=out/str(run_id)/str(a['id']);folder.mkdir(parents=True)
            # gh follows GitHub's signed artifact redirect without exposing tokens.
            raw=subprocess.check_output(['gh','api',f'repos/xrey167/FinX-BB/actions/artifacts/{a["id"]}/zip'])
            observed='sha256:'+hashlib.sha256(raw).hexdigest()
            if a.get('digest')!=observed:raise AssertionError(f'archive digest mismatch: {a["id"]}')
            ar['archive_sha256_verified']=observed;ar['files']=[]
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if info.is_dir():continue
                    if info.file_size>40_000_000:raise ValueError('oversized member')
                    data=z.read(info)
                    item={'path':info.filename,'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}
                    ar['files'].append(item)
                    if info.filename.endswith('.json'):
                        obj=json.loads(data)
                        # Flatten saved names; original archive paths stay in the index.
                        dest=folder/(str(len(ar['files']))+'-'+Path(info.filename).name)
                        dest.write_text(json.dumps(obj,indent=2)+'\n')
                        item['retained_path']=str(dest.relative_to(out))
                        if isinstance(obj,dict):
                            item['summary']={k:obj[k] for k in ('seed','model_name','experiment','screening_pass','metrics','criteria','fresh','attacks') if k in obj}
                            for k in ('fresh','reader'):
                                if k in obj:
                                    item['summary'][k]={kk:vv for kk,vv in obj[k].items() if kk!='rows'}
            ar['collection']='VERIFIED'
        index['runs'].append(entry)
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(index,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();collect(a.output)
