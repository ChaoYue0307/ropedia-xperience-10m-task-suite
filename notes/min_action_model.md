# Minimal Action Model

This is the first modeling baseline for the Ropedia/Xperience sample.

The script is:

```text
scripts/train_min_action_model.py
```

It trains a small Numpy-only softmax classifier:

```text
annotation.hdf5
  -> hand/body/IMU/camera/contact windows
  -> action or subtask labels from captions
  -> stratified train/test split
  -> multinomial logistic regression
  -> metrics and predictions
```

Run:

```bash
cd /path/to/Ropedia
source .venv/bin/activate
python scripts/train_min_action_model.py
```

Default output:

```text
outputs/min_action_model/
```

Important artifacts:

- `metrics.json`: accuracy, balanced accuracy, macro-F1, weighted-F1, majority baseline.
- `per_class_metrics.csv`: precision/recall/F1 per action class.
- `confusion_matrix.csv`: true label vs predicted label matrix.
- `predictions.csv`: one row per test window.
- `feature_dataset.npz`: processed numeric features and labels.
- `model.npz`: fitted scaler and softmax weights.

This is a learning baseline, not a publishable benchmark. The public sample is only one episode, so stratified windows from one episode are correlated. For serious evaluation, use many episodes and split by held-out episodes or held-out task instances.

## Current Sample Results

Action-label model:

```text
outputs/min_action_model/
accuracy:          0.9828
balanced_accuracy: 0.9644
macro_f1:          0.9688
weighted_f1:       0.9824
majority_baseline: 0.1375
classes:           18
test_windows:      291
```

Subtask-label model:

```text
outputs/min_subtask_model/
accuracy:          0.9759
balanced_accuracy: 0.9784
macro_f1:          0.9528
weighted_f1:       0.9779
majority_baseline: 0.1448
classes:           14
test_windows:      290
```

Why the numbers are high:

- This is one public sample episode.
- Windows are stratified randomly, so train/test windows can be close in time.
- The result proves the pipeline works; it does not prove cross-episode generalization.

Next serious evaluation:

```text
many episodes -> split by episode -> train on some episodes -> test on unseen episodes
```
