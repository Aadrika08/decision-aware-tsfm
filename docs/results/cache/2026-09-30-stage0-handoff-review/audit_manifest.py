"""Read-only metadata audit; no time-series arrays or forecasting models are read."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from unittest.mock import patch

from decision_aware_tsfm.cache.pilot import pilot_models
from decision_aware_tsfm.cache.planner import REQUIRED_ROW_FIELDS
from decision_aware_tsfm.cache.readiness import inspect_pilot

ROOT = Path(__file__).resolve().parent
inputs = ROOT/'inputs'
manifest = json.loads((inputs/'series_manifest.json').read_text())
omega_doc = json.loads((inputs/'omega_values.json').read_text())
active, excluded = manifest['active_dev_series'], manifest['excluded_series']
omega = omega_doc['omega_values']
omega_fields = ('omega_levels', 'omega_diffs', 'omega_levels_4096', 'omega_diffs_4096')

def inventory(rows):
    ids = Counter(r['series_id'] for r in rows)
    scoped = Counter((r['dataset'], r['config'], r['series_id']) for r in rows)
    identities = defaultdict(set)
    checksums = defaultdict(set)
    for r in rows:
        key = (r['dataset'], r['config'], r['item_id'], r['channel'])
        identities[key].add(r['series_id'])
        checksums[(r['dataset'], r['config'])].add(r['source_checksum'])
    return {
        'rows': len(rows), 'unique_ids': len(ids),
        'duplicate_id_extra_rows': sum(n-1 for n in ids.values()),
        'duplicate_scoped_id_extra_rows': sum(n-1 for n in scoped.values()),
        'native_identity_with_multiple_ids': sum(len(v)>1 for v in identities.values()),
        'fallback_item_ids': sum(r.get('item_id_is_fallback') is True for r in rows),
        'invalid_item_ids': sum(r.get('item_id') in (None, '', 'None', 'nan') for r in rows),
        'legacy_counter_ids': sum('_idx_' in r['series_id'] for r in rows),
        'univariate_null_channels': sum(r['channel'] is None for r in rows),
        'malformed_subset_checksums': sum(not re.fullmatch(r'sha256:[0-9a-f]{64}', r['source_checksum']) for r in rows),
        'configurations_with_multiple_checksums': sum(len(v)>1 for v in checksums.values()),
        'held_out_metadata_rows': sum(any(s in str(r.get(k, '')).lower() for s in ('m5', 'pjm') for k in ('dataset', 'config', 'subset')) for r in rows),
    }

active_ids = {r['series_id'] for r in active}
excluded_ids = {r['series_id'] for r in excluded}
oc = Counter(r['series_id'] for r in omega)
by_id = {r['series_id']: r for r in active}
cutoff_issues, count_issues, empty_rows, finite_shorter = [], [], [], 0
for o in omega:
    r = by_id.get(o['series_id'])
    if r is None:
        continue
    end = r['split_boundary_index']
    span = min(end, 4096)
    if o['history_start_index_4096'] != max(0, end-4096):
        cutoff_issues.append(o['series_id'])
    if not 0 <= o['history_length_4096'] <= span:
        count_issues.append(o['series_id'])
    if o['history_length_4096'] == 0:
        empty_rows.append({'series_id':o['series_id'], 'config':r['config'],
                           'history_start':o['history_start_index_4096'], 'history_end':end,
                           'history_length_4096':0, 'omega_levels_4096':o['omega_levels_4096'],
                           'omega_diffs_4096':o['omega_diffs_4096']})
    finite_shorter += o['history_length_4096'] < span

fields = set().union(*(r.keys() for r in active))
summary = {
    'active': inventory(active), 'excluded': inventory(excluded),
    'active_excluded_id_overlap': len(active_ids & excluded_ids),
    'active_config_counts': dict(sorted(Counter(r['config'] for r in active).items())),
    'dataset_values': sorted({r['dataset'] for r in active}),
    'active_rows_with_nonfinite_positions_in_trimmed_history': sum(r['length'] < r['split_boundary_index'] for r in active),
    'raw_trim_bound_violations': sum(r['split_boundary_index'] != max(0,r['raw_length']-r['test_trim']) or not 1000 <= r['length'] <= r['split_boundary_index'] for r in active),
    'omega': {
        'rows':len(omega), 'duplicate_id_extra_rows':sum(n-1 for n in oc.values()),
        'missing_active_ids':len(active_ids-set(oc)), 'unmatched_ids':len(set(oc)-active_ids),
        'value_checks': {k:{'nonfinite_or_out_of_range':sum(not isinstance(o[k],(float,int)) or not math.isfinite(o[k]) or not 0 <= o[k] <= 1 for o in omega),
                            'minimum':min(o[k] for o in omega), 'maximum':max(o[k] for o in omega)} for k in omega_fields},
        'recorded_4096_cutoff_mismatches':len(set(cutoff_issues)),
        'finite_count_outside_raw_span':len(set(count_issues)),
        'empty_4096_histories':len(empty_rows),
        'empty_4096_configs':dict(sorted(Counter(r['config'] for r in empty_rows).items())),
        'empty_4096_rows':empty_rows,
        'finite_count_less_than_raw_capped_span':finite_shorter,
        'raw_history_shorter_than_4096':sum(r['split_boundary_index']<4096 for r in active),
        'per_forecast_origin_rows':False,
        'interpretation':'Bounds checked against the series trim boundary; no Omega recomputation or per-origin leakage certification.'
    },
    'interface':{
        'manifest_top_level_keys': sorted(manifest),
        'active_row_fields':sorted(fields),
        'workload_required_fields_absent_by_name':sorted(REQUIRED_ROW_FIELDS-fields),
        'omega_top_level_keys':sorted(omega_doc),
        'omega_row_fields':sorted(set().union(*(r.keys() for r in omega))),
        'production_naming_header_present': 'naming_scheme' in manifest and 'source_repository' in manifest,
        'caution':'Absent-by-name fields are a mapping checklist, not evidence the Stage 0 inventory is invalid.'
    },
    'roster_observations':{
        'hospital_active_rows':sum('hospital' in r['config'].lower() for r in active),
        'jena_flat_active_rows':sum(r['config']=='jena_weather' for r in active),
        'us_births_active_rows':sum('us_births' in r['config'].lower() for r in active),
    },
    'limitations': ['Metadata audit only; source/subset hashes were not recomputed from raw arrays.',
                    'Stable mappings verified within this snapshot; Stage 0 was not regenerated or reordered.',
                    'No forecast-window schedule, approved roles/folds, or real consumer acceptance inferred.'],
}

with patch('decision_aware_tsfm.cache.readiness.load_source') as source_reader, \
     patch('decision_aware_tsfm.cache.executor.load_adapter') as adapter_loader:
    report, full, first = inspect_pilot(manifest, {'schema_version':1,'models':pilot_models()}, omega_doc, base_dir=inputs)
    assert report['status'] == 'blocked' and full is None and first is None
    source_reader.assert_not_called()
    adapter_loader.assert_not_called()
report['probe_kind'] = 'native_inventory_as_workload_admission'
report['interpretation'] = 'Expected shape rejection: a series inventory is not an executable workload. See the separate semantic mapping checklist.'
report['source_loader_calls'] = 0
report['adapter_loader_calls'] = 0
(ROOT/'native-readiness-probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(ROOT/'manifest-audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2,allow_nan=False))
