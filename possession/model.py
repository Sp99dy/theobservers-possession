"""v2 emission model: gradient boosting on degraded IDSSE features, and its training data.

Training rows come from several corrupted copies of each IDSSE match (independent degrader seeds)
and their team-swapped mirror images, so the classifier sees ATD-like inputs and cannot learn a
home-team bias. Rows are subsampled every SUBSAMPLE frames: neighbouring frames are near-duplicates.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, degrade, features, io_idsse
from .canonical import orient

# Training copies per match: (degrader seed offset, swapped). [D] two independent corruptions,
# each with its mirror image.
TRAIN_COPIES = ((0, False), (0, True), (1, False), (1, True))
TRAIN_COPIES_X2 = TRAIN_COPIES + ((2, False), (2, True), (3, False), (3, True))   # experiment
CLEAN_COPIES = ((None, False), (None, True))   # ablation only: no degrader (offset None = clean)
EVAL_COPY = (9, False)       # [D] a separate corruption, never trained on, for CV scoring
SUBSAMPLE = 5                # [D] one row per 0.2 s
HGB_PARAMS = dict(           # [D] conventional settings; early stopping off for determinism
    max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
    l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=0)


def copy_seed(match_index, offset):
    return 10_000 + 100 * match_index + offset


def make_copy(match_id, params, offset, swapped):
    idx = io_idsse.MATCH_IDS.index(match_id)
    clean = io_idsse.load_labelled(match_id)
    deg = clean if offset is None else degrade.degrade(clean, params, seed=copy_seed(idx, offset))
    if swapped:
        deg = degrade.swap_teams(deg)
    return orient(deg)


def _feature_cache(match_id, offset, swapped, version):
    return os.path.join(config.CACHE_DIR, "idsse_feat_{}_{}{}_v{}.npz".format(
        match_id, "clean" if offset is None else offset, "s" if swapped else "", version))


def copy_labels(match_id, swapped, control_min_s):
    """Labels of any copy of a match at bridge control_min_s. The degrader and orient() never
    change labels (frame count, D and X are kept), so they are the clean labels, team-swapped."""
    from .canonical import A, H
    lab = io_idsse.load_labelled(match_id, control_min_s).labels
    if swapped:
        lab = np.where(lab == H, A, np.where(lab == A, H, lab)).astype(np.int8)
    return lab


def _bridged(match_id, swapped, y_default):
    """Labels at the configured bridge; the default bridge returns the cached labels untouched."""
    if config.CONTROL_MIN_S == config.DEFAULT_CONTROL_MIN_S:
        return y_default
    return copy_labels(match_id, swapped, config.CONTROL_MIN_S)


def copy_features(match_id, params, offset, swapped, version, use_cache=True):
    """(X, labels, graded, names) for one corrupted copy, cached on disk (IDSSE is public data).
    The cache holds labels at the default bridge; other bridges are relabelled on load."""
    path = _feature_cache(match_id, offset, swapped, version)
    if use_cache and os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            return z["X"], _bridged(match_id, swapped, z["y"]), z["g"], z["names"].tolist()
    m = make_copy(match_id, params, offset, swapped)
    X, names = features.design_matrix(m, version=version)
    y_default = (m.labels if config.CONTROL_MIN_S == config.DEFAULT_CONTROL_MIN_S else
                 copy_labels(match_id, swapped, config.DEFAULT_CONTROL_MIN_S))
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, X=X, y=y_default, g=m.graded, names=np.array(names))
    os.replace(tmp, path)
    return X, m.labels, m.graded, names


def training_rows(match_ids, params, version, copies=TRAIN_COPIES):
    Xs, ys = [], []
    for mid in match_ids:
        for offset, swapped in copies:
            X, y, g, names = copy_features(mid, params, offset, swapped, version)
            idx = np.flatnonzero(g)[::SUBSAMPLE]
            Xs.append(X[idx])
            ys.append(y[idx])
    return np.vstack(Xs), np.concatenate(ys), names


def train(X, y, columns=None, **overrides):
    clf = HistGradientBoostingClassifier(**dict(HGB_PARAMS, **overrides))
    clf.fit(X if columns is None else X[:, columns], y)
    prior = np.bincount(y, minlength=3)[:3] / len(y)
    return {"clf": clf, "prior": prior, "columns": columns}


def predict_log_proba(model, X):
    Xc = X if model["columns"] is None else X[:, model["columns"]]
    p = model["clf"].predict_proba(Xc)
    out = np.full((len(X), 3), np.log(1e-6))
    for j, c in enumerate(model["clf"].classes_):
        out[:, int(c)] = np.log(np.clip(p[:, j], 1e-6, 1.0))
    return out


def emission_ll(model, X):
    """Scaled likelihoods log p(x|c) + const = log P(c|x) - log P(c), per frame.

    class_weight='balanced' already trains on equal class weights, so the classifier's output
    behaves like a posterior under a flat prior; we therefore subtract nothing further."""
    return predict_log_proba(model, X)


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=3)


def load(path):
    return joblib.load(path)
