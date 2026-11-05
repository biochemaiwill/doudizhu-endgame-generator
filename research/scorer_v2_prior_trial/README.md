# TabPFN v2 CPU validation and sealed test trial

TabPFN 2.0.9; sklearn 1.6.1 in isolated runtime, numpy/torch reused from research env.
Dependency installs used --target runtime --no-deps --no-cache-dir. Shared environment unchanged.
One public non-gated checkpoint: https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt (44.4 MB); checksum in validation.json.
License: https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt
No external inference service or data upload. CPU, one torch thread, 2 estimators per head, seed 73.
Run: /home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v2_prior_trial/run.py
Predict: same Python research/scorer_v2_prior_trial/predict.py input.json

During model selection only frozen train124/val12 were used. After parent selection_final.json was frozen, test17 inference was performed with unchanged settings; predictions_test.json records the outputs and no tuning occurred.
Three separately fitted targets: overall, reasoning_depth, move_concealment; output clipped1..4.
Original86 features: val macro MAE1.064642; structured143:0.535141, overall MAE0.421161.
This is not a class-weighted model: TabPFN fit accepts context points without sample weights in this version. No duplicated balancing points were added. Evaluate in unweighted candidate arm.
Validation predictions and exact train/val IDs preserved. Must freeze parent selection before any test evaluation.
runtime/ and *.ckpt are Git ignored to avoid committing vendor installation/weights. Required checkpoint can be downloaded and checksum verified; retain locally only if selected for continued use.
