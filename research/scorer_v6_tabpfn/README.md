# TabPFN v2 classifier group-CV candidate
Six fixed folds from scorer_v6/cv_split.json; only overall4-class target, argmax over class probabilities. No regression or rounding.
Mean CV accuracy0.5864169; grade4 recall0; not90%. Full189-point training context retained in context.json; load_model reconstructs exactly the classifier from context and public checkpoint. No strict holdout read.
Official checkpoint and SHA256 in model.json; license https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt
Isolated dependencies in scorer_v2_prior_trial/runtime: tabpfn2.0.9, sklearn1.6.1; CPU1 thread, estimators2, seed73. Weight29MB ignored by Git.
run.py reproduces CV. predict.py input.json reconstructs full-context classifier and returns class probabilities and overall argmax. Auxiliary heads are not trained or validated.
