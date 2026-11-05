# Fixed TabPFN classifier data-expansion comparison
Only training data changed; base143 features, TabPFN v2 classifier, estimators2, seed73, CPU1 thread unchanged.
Six validation ID sets are identical to scorer_v6/cv_split.json.
Expanded training IDs are exactly scorer_data300_v1/split_audit.json expanded_train_ids; family/source URL/all paired and parent relationships excluded if linked to validation.
Old result cited directly from scorer_v6_tabpfn/cv_results.json, with exact old train/val IDs checked.
Old mean CV accuracy:58.64169%; expanded:39.22294%; delta:-19.41875 percentage points.
Expanded class recalls1/2/3/4:74.07407%/85.68376%/0.87719%/0%.
No strict holdout labels/predictions were read. This is development CV, not external-test performance.
All predictions are class argmax over probabilities; no regression/rounding or retuning.
The degradation is retained; no sample removal or parameter selection followed it.
Runtime and checkpoint reused from scorer_v2_prior_trial/runtime and scorer_v6_tabpfn/classifier.ckpt. No new install/download.
Run with research Python: python research/scorer_data300_tabpfn/compare.py
Full fold predictions/IDs and config are in results.json; verification.json confirms file consistency.
