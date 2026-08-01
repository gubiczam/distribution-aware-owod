# Stage 2 T4 Smoke Protocol

Status: ready for real T4 smoke after local config validation. This does not start the 36-run campaign.

The smoke config is `configs/smoke_stage2_t4.yaml`: one strategy (`v2:uncertainty` with objectness-weighted entropy), one seed, one round, budget 2, one training epoch. It uses the same checkpoint, same OWDETR/SOWODB flags, same train/predict/evaluate bridge, same grouped metrics, same output persistence, and same completed-round overwrite protection as the final configs.

## Colab Cells

Cell 1 - mount Drive and clone exact repos:

```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!test -d distribution-aware-owod || git clone https://github.com/gubiczam/distribution-aware-owod.git distribution-aware-owod
!test -d PROB || git clone https://github.com/gubiczam/PROB.git PROB
```

Cell 2 - verify the source revision policy:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
git fetch origin
git status --short
test -z "$(git status --short)" || { echo "FAIL: commit and push local Stage 2 protocol changes before smoke"; exit 2; }
cd /content/drive/MyDrive/PROB
git fetch origin
git checkout 980cf3a796f064dd4c56f573ba10cc755143e116
mkdir -p /Users/gubiczam/Documents /Users/gubiczam/Downloads/results
ln -sfn /content/drive/MyDrive/PROB /Users/gubiczam/Documents/PROB
ln -sfn /content/drive/MyDrive/owod_stage /Users/gubiczam/owod_stage
ln -sfn /content/drive/MyDrive/results/SOWODB /Users/gubiczam/Downloads/results/SOWODB
```

Cell 3 - install DAOWOD dependencies:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Cell 4 - install/compile PROB dependencies and CUDA extension:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/PROB
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Cell 5 - validate assets and hashes:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
test -d /content/drive/MyDrive/owod_stage
test -d /content/drive/MyDrive/results/SOWODB
test -f outputs/stage1b/stage1b_candidate_500.txt
test -f outputs/stage1b/stage1b_reference_3500.txt
test -f /Users/gubiczam/owod_stage/ImageSets/OWDETR/owdetr_test.txt
test -d /Users/gubiczam/owod_stage/Annotations
test -f /Users/gubiczam/Downloads/results/SOWODB/t1.pth
sha256sum /Users/gubiczam/Downloads/results/SOWODB/t1.pth | grep dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22
PYTHONPATH=src python analysis/stage2_plan.py
python - <<'PY'
import json
preflight=json.load(open('outputs/stage2_plan/protocol_preflight.json'))
assert preflight['asset_status']=='ready', preflight
assert preflight['protocol_status']=='ready', preflight
assert preflight['candidate_evaluation_overlap']==0, preflight
assert preflight['reference_evaluation_overlap']==0, preflight
PY
```

Cell 6 - compile and test DAOWOD:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
python -m compileall src analysis tests
ruff format --check .
ruff check .
pytest
```

Cell 7 - validate all final configs plus smoke and write a machine-readable preflight:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
mkdir -p outputs/stage2_smoke_t4
set +e
for cfg in configs/stage2_*.yaml configs/smoke_stage2_t4.yaml; do
  daowod-run validate --config "$cfg" --manifest "outputs/stage2_plan/$(basename "$cfg" .yaml)_validate_manifest.json"
done
code=$?
set -e
python - "$code" <<'PY'
import json, sys
code=int(sys.argv[1])
summary={
  "schema":"stage2_t4_smoke_preflight_v1",
  "verdict":"PASS" if code == 0 else "FAIL",
  "stage":"config_validation",
  "exit_code":code,
  "expected_current_failure":"None"
}
open("outputs/stage2_smoke_t4/preflight_summary.json","w").write(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY
```

Cell 8 - run the smoke experiment only if validation passed:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json, sys
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
if summary["verdict"] != "PASS":
    print("Skipping T4 smoke because preflight failed:")
    print(json.dumps(summary, indent=2))
    sys.exit(1)
PY
then
daowod-run campaign --config configs/smoke_stage2_t4.yaml
else
  echo "T4 smoke skipped after preflight failure"
fi
```

Cell 9 - check artifacts, resolved commands, grouped metrics, and resume refusal:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json, sys
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
if summary["verdict"] != "PASS":
    print("Skipping artifact checks because smoke did not run:")
    print(json.dumps(summary, indent=2))
    sys.exit(1)
PY
then
MANIFEST=$(find outputs/stage2_smoke_t4 -name round_manifest.json | sort | tail -1)
python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
m=json.load(open(sys.argv[1]))
round_dir=Path(sys.argv[1]).parent
assert m['completed'] is True
assert m['resolved_command_parity']['status']=='ok'
assert m['evaluation_count'] > 0
assert m['support_counts']['evaluation']['head_objects'] > 0
assert m['support_counts']['evaluation']['medium_objects'] > 0
assert m['support_counts']['evaluation']['tail_objects'] > 0
assert m['grouped_metrics'] is not None
for name in ['candidate_ids_before_selection.txt','reference_ids.txt','labelled_ids_before_selection.txt','selected_ids.txt','labelled_ids.txt','training_ids.txt','remaining_pool_ids.txt','evaluation_ids.txt','metrics.json','checkpoint.pth']:
    assert (round_dir/name).exists(), name
print('artifact check PASS')
PY
set +e
daowod-run campaign --config configs/smoke_stage2_t4.yaml
code=$?
set -e
test "$code" -ne 0
echo "resume refusal PASS"
else
  echo "artifact checks skipped after preflight failure"
fi
```

Cell 10 - final PASS/FAIL:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
raise SystemExit(0 if summary["verdict"] == "PASS" else 1)
PY
then
test -f outputs/stage2_smoke_t4/selections.json
test -f outputs/stage2_smoke_t4/metrics.csv
echo "STAGE2_T4_SMOKE_PASS"
else
cat outputs/stage2_smoke_t4/preflight_summary.json
echo "STAGE2_T4_SMOKE_FAIL"
fi
```
