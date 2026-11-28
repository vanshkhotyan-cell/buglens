# Synchronized Rollback Runbook

1. **Container Rollback**: Redeploy the previous immutable GHCR digest (ghcr.io/vanshkhotyan-cell/buglens@sha256:<previous-digest>).
2. **Model Alias Rollback**: Reassign @champion in MLflow Model Registry via MlflowClient().set_registered_model_alias("buglens", "champion", "<previous-version>") and set BUGLENS_MODEL_URI=models:/buglens/<previous-version>.
3. **Code & Data Rollback**: Check out the target release commit (git checkout <commit-or-tag>) and restore exact dataset artifacts (python -m dvc pull).
