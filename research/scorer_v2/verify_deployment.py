"""End-to-end deployment check against the sealed test prediction artifact."""
import json,hashlib
from pathlib import Path
import numpy as np
from predict import predict_spec
ROOT=Path(__file__).resolve().parent
rows={r["id"]:r for r in map(json.loads,(ROOT/"dataset.jsonl").read_text().splitlines())}
saved=json.loads((ROOT/"run/predictions_test_final.json").read_text())
model=json.loads((ROOT/"run/model.json").read_text())
p=predict_spec(model["spec"],[rows[r["id"]]["input"] for r in saved])
expected=np.array([r["prediction"] for r in saved])
delta=float(np.abs(p-expected).max())
assert p.shape==(17,3) and np.isfinite(p).all()
assert np.all((p>=1)&(p<=4)) and np.allclose(p,expected,atol=3e-5)
out={"rows":17,"max_absolute_difference":delta,"portable_predictor_matches_sealed_predictions":True,"test_used_to_change_model":False}
(ROOT/"run/deployment_verification.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
