Change the **model or its objective**. This is the lever the organisers rate highest:
the metrics are ranking metrics but the baseline optimises pointwise logloss.

`lightgbm.LGBMRanker` with `objective="lambdarank"` and `group` set to per-user
impression counts matches the task structure directly. If you use it, remember the
group array must follow the row order of the frame you train on, so sort *training*
rows by user (never the target frame) and build groups from the sorted counts.

Keep the model CPU-cheap: this must train and predict inside the iteration timeout.
