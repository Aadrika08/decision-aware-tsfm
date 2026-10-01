"""Probe unchanged consumers using only synthetic partitions and negative controls."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
bundle = ROOT/'consumer-fixture'
handoff = json.loads((bundle/'handoff.json').read_text())
hashes = handoff['partition_hashes']
for relative, sha in hashes.items():
    assert hashlib.sha256((bundle/relative).read_bytes()).hexdigest() == sha

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('decision','k3'))
mode = parser.parse_args().mode

if mode == 'decision':
    from decision_aware_tsfm.analysis.decision.cache_io import load_cache
    frame = pa.concat_tables([pq.ParquetFile(bundle/p).read().replace_schema_metadata(None) for p in hashes]).to_pandas()
    results = {'input_rows':len(frame), 'partitions':len(hashes), 'probes':[]}
    for name, data, version in (
        ('native_producer_schema_v1',frame,1),
        ('diagnostic_rename_only_preserving_all_seeds',frame.rename(columns={'y_true':'y'}),'fixture-v0'),
    ):
        try:
            loaded = load_cache(data, schema_version=version)
            results['probes'].append({'name':name,'outcome':'accepted','rows':len(loaded)})
        except Exception as e:
            results['probes'].append({'name':name,'outcome':'rejected','error_type':type(e).__name__,'message':str(e)})
else:
    from decision_aware_tsfm.analysis.k3.loader import load_k3_inputs
    results = {'fixture':{},'negative_controls':[],
               'negative_control_inputs':'Invented data only; any consumer status=ok is a demonstrated validation gap, not real acceptance.'}
    for variant in ('current','last_4096'):
        _,_,report = load_k3_inputs(bundle,hashes,bundle/'omega.json',variant,allow_fixture=True)
        results['fixture'][variant]=report

    # Repository header mismatch on the unchanged full synthetic bundle.
    wrong = json.loads((bundle/'omega.json').read_text())
    wrong['source_repository'] = 'different/invented-repository'
    omega_path=ROOT/'mismatched-repository-omega.json'
    omega_path.write_text(json.dumps(wrong)+'\n')
    try:
        _,_,report=load_k3_inputs(bundle,hashes,omega_path,'current',allow_fixture=True)
        results['negative_controls'].append({'name':'repository_header_mismatch','expected':'reject','observed':report['status']})
    except Exception as e:
        results['negative_controls'].append({'name':'repository_header_mismatch','expected':'reject','observed':'rejected','error':str(e)})

    # Select one complete three-model comparison, remove provenance, and present
    # it to the non-fixture admission path solely as a negative validation test.
    keys=('dataset','config','series_id','origin','horizon','seed')
    selected,chosen=[],None
    for relative in hashes:
        table=pq.ParquetFile(bundle/relative).read()
        row=table.slice(0,1).to_pylist()[0]
        key=tuple(row[k] for k in keys)
        if chosen is None:chosen=key
        if key==chosen:selected.append(table)
    assert {t['family'][0].as_py() for t in selected} == {'classical','tsfm'}
    negative=ROOT/'negative-control-missing-provenance'
    negative.mkdir(exist_ok=False)
    negative_hashes={}
    for i,table in enumerate(selected):
        path=negative/f'part-{i}.parquet'
        pq.write_table(table.replace_schema_metadata(None),path)
        negative_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    doc=json.loads((bundle/'omega.json').read_text())
    doc['fixture']=False
    doc['rows']=[r for r in doc['rows'] if tuple(r[k] for k in keys[:4])==chosen[:4]]
    path=negative/'omega.json';path.write_text(json.dumps(doc)+'\n')
    try:
        _,_,report=load_k3_inputs(negative,negative_hashes,path,'current',allow_fixture=False)
        results['negative_controls'].append({'name':'missing_real_job_metadata','expected':'reject','observed':report['status'],
                                             'partitions':report['partitions'],'provenance_checked_jobs':report['provenance_checked_jobs']})
    except Exception as e:
        results['negative_controls'].append({'name':'missing_real_job_metadata','expected':'reject','observed':'rejected','error':str(e)})

(ROOT/f'{mode}-probe.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps(results,indent=2,allow_nan=False))
