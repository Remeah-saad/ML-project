"""
model_logistic_regression.py
=============================
Model 1: TF-IDF Vectorisation + Logistic Regression

This is the primary model described in the project hypothesis:
  "If we apply appropriate text preprocessing techniques and use machine
   learning algorithms such as Logistic Regression or Naive Bayes, then
   the model will be able to classify Google Play reviews into positive and
   negative sentiments with high accuracy."

Architecture:
  Arabic cleaned text
       ↓
  TF-IDF Vectoriser  (character n-grams + word n-grams, Arabic-aware)
       ↓
  Logistic Regression  (class_weight='balanced' for imbalance – spec §5)

Also includes Naive Bayes as a secondary comparison within this module
(per research hypothesis mentioning both LR and NB).

Outputs:
  - models/model_lr.joblib
  - models/model_nb.joblib
  - outputs/lr_confusion_matrix.png
  - outputs/lr_top_features.png
"""

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # Arabic may need specific font

import seaborn as sns

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

log = logging.getLogger(__name__)

LABEL_NAMES = ["negative", "positive"]


# ── TF-IDF configuration for Arabic ──────────────────────────────────────────
#
# Arabic-specific choices:
#   - analyzer='char_wb': character n-grams handle morphological richness of Arabic
#     (prefixes, suffixes, clitics). Works alongside word n-grams.
#   - sublinear_tf=True: dampens frequency counts – important since Arabic
#     words repeat frequently with different morphological forms
#   - min_df=2: removes hapax legomena (noisy reviews, spelling errors §5)

def _make_tfidf_word() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents=None,   # do NOT strip – Arabic script must stay intact
        lowercase=False,      # Arabic has no case
    )


def _make_tfidf_char() -> TfidfVectorizer:
    """Character n-gram TF-IDF – handles Arabic morphology better."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
    )


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_lr_pipeline(use_char_ngram: bool = False) -> Pipeline:
    """
    TF-IDF → Logistic Regression.
    class_weight='balanced' addresses class imbalance (spec §5 Imbalanced Data bias).
    """
    tfidf = _make_tfidf_char() if use_char_ngram else _make_tfidf_word()
    return Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )),
    ])


def build_nb_pipeline() -> Pipeline:
    """
    TF-IDF → Complement Naive Bayes.
    ComplementNB performs better than MultinomialNB on imbalanced text data.
    Included per hypothesis mention of Naive Bayes.
    """
    return Pipeline([
        ("tfidf", _make_tfidf_word()),
        ("clf", ComplementNB(alpha=0.5)),
    ])


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

LR_PARAM_GRID = {
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "tfidf__max_features": [50_000, 80_000],
    "clf__C":              [0.01, 0.1, 1.0, 10.0],
}


def tune_pipeline(
    pipeline: Pipeline,
    X_train: list,
    y_train: np.ndarray,
    param_grid: dict,
    cv: int = 5,
) -> Pipeline:
    log.info(f"Running GridSearchCV (cv={cv}, scoring=f1_macro) …")
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    log.info(f"Best params : {gs.best_params_}")
    log.info(f"Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    text_col:   str  = "cleaned_content",
    label_col:  str  = "binary_label",
    model_name: str  = "lr",          # 'lr' | 'nb'
    tune:       bool = False,
    save_dir:   Optional[str] = "models",
) -> Pipeline:
    """
    Train and validate Model 1.
    Returns the fitted sklearn Pipeline.
    """
    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].values
    X_val   = val_df[text_col].tolist()
    y_val   = val_df[label_col].values

    pipeline = build_lr_pipeline() if model_name == "lr" else build_nb_pipeline()

    if tune and model_name == "lr":
        pipeline = tune_pipeline(pipeline, X_train, y_train, LR_PARAM_GRID)
    else:
        log.info(f"Fitting {model_name.upper()} pipeline …")
        pipeline.fit(X_train, y_train)

    # Validation report
    y_pred_val = pipeline.predict(X_val)
    log.info(f"\nValidation results ({model_name.upper()}):")
    log.info(f"\n{classification_report(y_val, y_pred_val, target_names=LABEL_NAMES)}")

    # 5-fold cross-validation on train set
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro", n_jobs=-1,
    )
    log.info(f"5-fold CV F1 (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / f"model_{model_name}.joblib"
        joblib.dump(pipeline, path)
        log.info(f"Saved → {path}")

    return pipeline


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pipeline:    Pipeline,
    test_df:     pd.DataFrame,
    text_col:    str = "cleaned_content",
    label_col:   str = "binary_label",
    model_name:  str = "lr",
    plot_dir:    Optional[str] = "outputs",
) -> Dict[str, Any]:
    X_test = test_df[text_col].tolist()
    y_test = test_df[label_col].values

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")
    auc  = roc_auc_score(y_test, y_prob)

    log.info(f"\n{'='*50}")
    log.info(f"  Model 1 ({model_name.upper()}) – Test Set Results")
    log.info(f"{'='*50}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  Macro F1  : {f1:.4f}")
    log.info(f"  ROC-AUC   : {auc:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=LABEL_NAMES)}")

    # Confusion matrix
    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(y_test, y_pred, model_name, plot_dir)

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1_macro":  f1,
        "roc_auc":   auc,
        "report":    classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True),
    }


def _plot_confusion_matrix(y_true, y_pred, model_name: str, plot_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Model 1 ({model_name.upper()}) – Confusion Matrix\n(Arabic Google Play Reviews)")
    plt.tight_layout()
    out = f"{plot_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Confusion matrix → {out}")


# ── Feature importance ────────────────────────────────────────────────────────

def plot_top_features(
    pipeline: Pipeline,
    n: int = 20,
    plot_dir: Optional[str] = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts top positive and negative n-gram features from LR coefficients.
    Useful for understanding what Arabic terms drive sentiment decisions.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        log.warning("Feature importance only available for Logistic Regression.")
        return pd.DataFrame(), pd.DataFrame()

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[::-1][:n]
    top_neg_idx = np.argsort(coefs)[:n]

    pos_df = pd.DataFrame({"feature": feature_names[top_pos_idx], "coefficient": coefs[top_pos_idx]})
    neg_df = pd.DataFrame({"feature": feature_names[top_neg_idx], "coefficient": coefs[top_neg_idx]})

    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].barh(pos_df["feature"], pos_df["coefficient"], color="#2ecc71")
        axes[0].set_title("Top Positive Features")
        axes[0].invert_yaxis()

        axes[1].barh(neg_df["feature"], neg_df["coefficient"], color="#e74c3c")
        axes[1].set_title("Top Negative Features")
        axes[1].invert_yaxis()

        plt.suptitle("Model 1 (LR) – Most Influential Arabic N-grams", fontsize=13)
        plt.tight_layout()
        out = f"{plot_dir}/lr_top_features.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Feature importance chart → {out}")

    return pos_df, neg_df


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(pipeline: Pipeline, texts: list) -> list:
    """Returns 'positive' or 'negative' for each text."""
    preds = pipeline.predict(texts)
    return [LABEL_NAMES[p] for p in preds]


def predict_proba(pipeline: Pipeline, texts: list) -> np.ndarray:
    """Returns probability array [P(negative), P(positive)]."""
    return pipeline.predict_proba(texts)
