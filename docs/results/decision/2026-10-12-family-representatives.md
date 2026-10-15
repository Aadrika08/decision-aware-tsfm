# Decision engine: choosing and freezing one representative model per family

Status: new code, tested on synthetic caches only. No real cache has been read. Not a result.
Scope: `analysis/decision/representatives.py` and `run_representatives.py` (new), one added builder
in `analysis/decision/fixture.py`, `tests/unit/decision/test_decision_representatives.py` (new).
Issue #41 (part of #36). The engine itself is unchanged: it still takes a `{family: model}` dict,
and this change only produces that dict.

**The rule has to be recorded in `docs/decisions/` before any real result is seen.** Nothing here
decides it. The code implements both rules from #41 so the team can pick one and keep the other as
the robustness check. Until the team has recorded the choice, do not run the command on the real
pilot cache and then look at engine results.

## Why now

The engine compares one model per family. The pilot roster has three classical models and one TSFM,
so the engine cannot run on the pilot until something picks the classical representative.

## Command

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

```bash
PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_representatives BUNDLE \
    --variant current --out artifacts/runs/representatives/representatives.json
```

`BUNDLE` is a producer handoff bundle (read with every partition hash checked) or a producer-v1
parquet file or directory. The command prints the score table and writes the frozen file. Run again
with the same input and the same `--out`, it reproduces the file and leaves it alone; if the result
would differ it stops and never overwrites. Options: `--rule fixed_named --classical MODEL --tsfm
MODEL` names the representatives, `--loss raw` averages unscaled loss, `--val-fraction` (default one
third), `--fixture` uses the synthetic roster fixture.

The fixture file below is reproduced by:

```bash
PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_representatives --fixture \
    --out artifacts/runs/representatives_fixture/representatives.json
```

From Python: `load_representatives(path)` returns the frozen dict for `build_tables` and `run`.

## The rule, exactly

`best_of_family` ("mean median pinball loss on development data", sprint task 4):

1. Roles. `costs.assign_roles` on every origin in the cache, the same function the engine uses: per
   (series, horizon) the earliest third of origins is validation, less the origins the gap rule
   removes, and the rest is evaluation (`test` in the code). Every row that is not validation is
   dropped before any loss is computed.
2. Unit loss. Per model and per unit (dataset, config, series_id, horizon): the mean pinball loss at
   tau 0.50 over the unit's validation targets, seeds averaged per window as the engine does, and
   the unit's scale, mean |y| over the same targets. These are the engine's `J` and `scale` from
   `costs.unit_costs`, taken on validation origins instead of evaluation origins.
3. Scaled by default. A unit counts `J / scale`. Reason: the engine never averages raw cost across
   series (scaled regret, and `best_fixed` sums `J / scale`), and a raw mean is decided by whichever
   dataset has the largest numbers. The scale uses validation targets only, so no evaluation value is
   read. `--loss raw` gives the unscaled reading. Both means are written to the file for every
   candidate, together with the model each one would pick, whichever is used.
4. Coverage. A model covers a unit only if it has every target (origin, step) that any model in the
   cache has for that unit. A family's models are compared on the units all of them cover, so a model
   cannot win by missing a hard series or horizon. Units with validation scale 0 are counted and left
   out. Coverage per model and the number of units scored are in the file.
5. Score. The plain mean of the unit losses over those common units. Every unit weighs the same,
   whatever its number of origins or forecast steps.
6. Choice and tie rule. Lowest score wins. Scores within 1e-12 relative of the lowest (the engine's
   `TIE_RTOL`) are a tie, and a tie goes to the model whose name sorts first. A family with one model
   returns it with the reason `only_member`.

`fixed_named`: the caller names both models. They are passed to `cache_io.restrict_to_representatives`
on the same cache, so whatever the engine would refuse is refused with the same message. The
validation scores of every candidate are still written, so the two rules can be compared.

Guards: dataset or config names matching the held-out rule (M5, PJM) are refused before anything is
scored, using `guards.assert_no_held_out`. The scoring function refuses any frame that holds a row
whose role is not validation. A frozen file that names held-out input is refused by the loader.

## The frozen file

JSON, UTF-8, LF, trailing newline, in the same `{"payload": ..., "sha256": ...}` layout as E2's frozen
specification, and read with E2's own check. It holds: the rule, the chosen model per family with
the reason, the aggregation and tie rule in words, the full score table (scaled and raw mean per
candidate, distance to the best, units covered, units scored), coverage counts per family, origins
per role, the validation fraction, the statement that losses were computed on validation origins
only, a fingerprint of the input rows (independent of row order), and a fingerprint of the selecting
code (`representatives.py` and `costs.py`). The loader refuses an edited, truncated or incomplete
file. As in E2 the hash guards against accidental edits; it is not a security control. The loader
does not refuse when the code fingerprint differs from the current code, because it does not re-run
the selection; re-running the command is what shows a difference.

## Robustness helper

`representative_robustness(...)` runs the existing `build_tables` and `run` once per classical
candidate with the TSFM side fixed and stacks `evaluate.summarize`. It reads evaluation origins, so
it is an analysis of results: it is for after the rule and the representatives are frozen, never for
choosing them.

## Verified in this change

- Decision suite: 199 tests (198 pass, 1 skipped without `DECISION_HANDOFF_BUNDLE`), up from 152 by
  the 47 new tests. Experiments 169, E3 107, K2 182, package layout 3, integration 102 (17 skipped):
  all unchanged in count and all pass. Same on Python 3.10.
- Default fixture run (`run_fixture`) is byte-identical to main on both Pythons: SHA-256 prefixes
  `fixture_cache.parquet bcecefbdb163`, `fixture_omega.csv 6e8679fa77c0`, `planted_answers.csv
  bd92190ccef0`, `selection_results.parquet 63c5dfb8911d`, `summary_by_dataset.csv cc17666045f9`.
- The frozen file from the fixture command is byte-identical across two runs on each of Python 3.11
  and 3.10 (4 files, 4,422 bytes, file SHA-256 prefix `7b2f710afec5`), and a rerun onto the existing
  file reports it reproduced and leaves it unchanged.
- A test builds a cache where the model that is best on validation origins is not the model that is
  best on evaluation origins or overall, checks that the three windows really disagree, and checks
  that the validation one is chosen. Another replaces every evaluation forecast and checks that no
  score changes.
- On the roster fixture each candidate's score equals the mean of the engine's own validation costs
  (`build_tables(...)["val"]` at tau 0.50) for that representative, so the window scored is the
  engine's validation window.
- 33 deliberately planted bugs, one at a time in a throwaway copy, are each caught by the tests.
  They include: scoring evaluation origins, scoring every origin, scoring gap origins, argmax for
  argmin, tie rule reversed, tie tolerance removed, coverage ignored, held-out check removed (from
  the selection and from the loader), hash check removed, wrong tau (0.40, 0.60, all taus), mean
  over rows instead of units, raw loss as default, scale not applied. The first pass caught 30 of
  33. The three misses showed that the held-out refusal was only tested through a check that runs
  after scoring, and that the fingerprint's independence from row order was not tested directly;
  two tests were added and all 33 are now caught.

## Fixture run: SYNTHETIC, not a result

Roster fixture: the existing synthetic fixture plus a third classical model (`AutoARIMA`, bias 0.8
and spread 1.2 in every dataset), so three classical candidates and one TSFM. 163,200 forecast rows,
5 datasets, 34 series, horizons 1 and 24, one seed; 5,780 validation origins, 11,560 evaluation
origins, no gap origins. All four models cover all 68 (series, horizon) units.

| family | model | chosen | mean scaled loss | mean raw loss | minus best |
|---|---|---|---|---|---|
| classical | AutoARIMA | yes | 0.052187 | 0.521499 | 0.000000 |
| classical | AutoETS | no | 0.054090 | 0.540123 | 0.001903 |
| classical | SeasonalNaive | no | 0.101470 | 1.014014 | 0.049283 |
| tsfm | Moirai-small | yes (only member) | 0.053233 | 0.532300 | 0.000000 |

These numbers say nothing about real models: the fixture's forecasts are drawn from planted
formulas, and the names are labels. What the run shows is that the command works end to end and that
the runner-up is visible in the file. Scaled and raw loss pick the same model here.

Robustness helper on the same fixture (mean scaled regret per selector, 484 units each, synthetic):
with `AutoARIMA`, `AutoETS` and `SeasonalNaive` as the classical representative, `static_omega`
gives 0.00098, 0.00039 and 0.00668, and `always_tsfm` gives 0.00354, 0.00541 and 0.00019. So on this
fixture the headline moves with the representative, which is what the helper is for.

## For the team to decide

1. **Which rule is the primary: best-of-family, or a fixed named representative with best-of-family
   as the robustness check.** Best-of-family gives each family its strongest member on the
   development sample, but the comparison then depends on the roster and on selection noise. A fixed
   name is declared up front and cannot be tuned, but may handicap a family. Issue #41 notes that in
   K2 the published effect disappeared under best-of-family, so this choice can change the headline.
2. **Raw or scaled loss.** Built: scaled by default, raw as an option, both recorded. The sprint text
   says "mean median pinball loss" without saying which. Raw lets the largest-valued series choose
   the representative; scaled treats every unit alike but divides by mean |y|, which is unstable for
   series near zero.
3. **One representative overall, one per horizon, or one chosen inside each leave-one-dataset-out
   fold.** Built: one overall, because the engine takes one dict per run. Per horizon would let the
   best classical model differ between horizon 1 and 168, at the price of one engine run per horizon
   and a family that means different models in different tables. Per fold would keep the evaluated
   dataset's own validation origins out of its representative choice, which is stricter than what is
   built, but gives one representative per dataset and nothing single to carry to M5 and PJM.

Choices made in this build that need a yes or a correction:

- Every unit weighs the same in the score. A dataset with more series therefore has more say.
  Alternative: average within each dataset first.
- A tie goes to the model whose name sorts first. It is deterministic but arbitrary.
- A unit is compared only if every candidate has every target there. One missing forecast drops the
  unit for the whole family; the counts in the file show how often.
- Fit-only datasets, if any are used, are treated like the others: their earliest third is scored.
- The held-out check covers dataset and config names, as in `pipeline.build_tables`; series IDs are
  not checked.

## Open

- Not run on any real cache. The first real run is the pilot cache, after the rule is recorded.
- The engine drops units whose evaluation scale is 0. The selection cannot know that without reading
  evaluation values, so such a unit can count in the choice and then be absent from the evaluation.
- `representatives.py` imports three private helpers from `analysis/experiments/e2.py` so both
  frozen files share one format and one check. Moving them to a shared module would touch E2 and is
  left for the owner's call.
- With one TSFM in the pilot the TSFM side is always `only_member`. A within-TSFM choice needs the
  expansion round (#41 "Depends on").
