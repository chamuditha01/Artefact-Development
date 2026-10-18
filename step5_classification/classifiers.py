"""
SCOPE NOTE (decisions.md, 2026-09-29 "even later" entry): the windows this
classifier's training features (claude/features_REF_merged.csv) were
extracted with - feature_extraction_v1.js's fixed T1=12mo/T2a=60d/T2b=12mo,
one classification per event - do NOT match decisions.md's later, real
design (T1=6mo, T2a=90-day floor, T2b=12-month floor, quarterly recheck
after the floor). This is a known, accepted gap for the 2026-10-20
submission, not an oversight to silently fix here - see that entry before
touching window lengths or re-run cadence anywhere in this file.

classifiers.py

Stage 1 (human vs natural) and Stage 2 (cultivation vs permanent_encroachment)
classifiers, as decided in decisions.md:
  - Stage 1: hand-set thresholds, unanimous 3-of-3 vote (2026-09-28, "later
    still, again"). NOT trained - REF has only 1 confirmed natural event.
  - Stage 2: logistic regression on 3 standardized features, chosen over a
    higher-LOOCV-scoring decision stump on small-sample overfitting grounds
    (2026-09-29). Deployed model is fit on ALL 19 labelled Stage-2-eligible
    events - LOOCV (stage2_loocv.py) is for the accuracy claim only, this
    module is for the classifier that actually runs.

WHAT THIS DOES NOT DO (stated per decisions.md 2026-09-28 "later" entry):
  Stage 2 does not attempt automatic `uncertain` detection. It always
  returns a cultivation-or-permanent_encroachment lean. `uncertain` stays
  a label a human writes, never one this code assigns.

Stage 2 confidence band (decisions.md, 2026-09-29 "later" entry - "Stage 2
confidence-band rule settled"): the probability output maps to a
`confidence` string ("high"/"medium"/"low", the field ClassificationState
requires) via a fixed, round, stated-in-advance margin rule (see
`_confidence_band`), not one calibrated from the small LOOCV sample -
mirroring the precedent set by Stage 1's percentile thresholds
(decisions.md, 2026-09-28). The numbers are an unvalidated placement
choice, not a measured reliability tier; see that entry's "What I gave
up" and "Revisit if" before quoting a Stage 2 confidence label as
meaningful on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Stage 1: hand-set thresholds (decisions.md, 2026-09-28 "later still, again") ---

STAGE1_THRESHOLDS = {
    "shape_compactness": 0.36,     # candidate_natural requires <  this
    "slope_deg": 4.43,              # candidate_natural requires >  this
    "dist_to_prior_loss_m": 212.0,  # candidate_natural requires >  this
}


@dataclass
class Stage1Result:
    label: str              # "candidate_human" | "candidate_natural"
    legs_passed: dict       # {"shape_compactness": bool, "slope_deg": bool, "dist_to_prior_loss_m": bool}
    confidence: str
    evidence_note: str


def stage1_classify(features: dict) -> Stage1Result:
    """
    features must carry: shape_compactness, slope_deg, dist_to_prior_loss_m.
    Unanimous (3-of-3) vote required for candidate_natural - a 2-of-3 vote
    was tested and rejected (decisions.md) because it pulled two real
    cultivation events (REF_052, REF_025) out of Stage 2's pool.
    """
    legs = {
        "shape_compactness": features["shape_compactness"] < STAGE1_THRESHOLDS["shape_compactness"],
        "slope_deg": features["slope_deg"] > STAGE1_THRESHOLDS["slope_deg"],
        "dist_to_prior_loss_m": features["dist_to_prior_loss_m"] > STAGE1_THRESHOLDS["dist_to_prior_loss_m"],
    }
    label = "candidate_natural" if all(legs.values()) else "candidate_human"

    # Confidence: this is a deterministic threshold rule, not a probabilistic
    # model, so "high" describes certainty that the RULE was applied
    # correctly - not a claim about Stage 1's real-world accuracy, which
    # decisions.md already flags as "indicative only" for the natural class
    # (n=1) until more natural events are labelled.
    note = (
        f"compactness={features['shape_compactness']:.3f} "
        f"({'<' if legs['shape_compactness'] else '>='} {STAGE1_THRESHOLDS['shape_compactness']}); "
        f"slope={features['slope_deg']:.2f} "
        f"({'>' if legs['slope_deg'] else '<='} {STAGE1_THRESHOLDS['slope_deg']}); "
        f"dist_prior_loss={features['dist_to_prior_loss_m']:.1f} "
        f"({'>' if legs['dist_to_prior_loss_m'] else '<='} {STAGE1_THRESHOLDS['dist_to_prior_loss_m']}); "
        f"{sum(legs.values())}/3 legs passed, 3/3 required for candidate_natural"
    )
    return Stage1Result(label=label, legs_passed=legs, confidence="high", evidence_note=note)


# --- Stage 2: logistic regression (decisions.md, 2026-09-29) -----------------

STAGE2_FEATURES = ["trend_nbr_per_day", "wobble_std", "cycle_count_provisional"]


@dataclass
class Stage2Result:
    label: str                  # "cultivation" | "permanent_encroachment"
    probability_permanent: float
    confidence: str
    evidence_note: str


def _confidence_band(prob_permanent: float) -> str:
    """Fixed margin bands on predict_proba's distance from 0.5 - logged
    decision, decisions.md 2026-09-29 "later" entry ("Stage 2
    confidence-band rule settled"). Not calibrated from the LOOCV sample."""
    margin = abs(prob_permanent - 0.5)
    if margin >= 0.3:
        return "high"
    if margin >= 0.15:
        return "medium"
    return "low"


# Also export band() as an alias for e1_recheck_and_extras.py compatibility
band = _confidence_band


class Stage2Classifier:
    """Logistic regression on 3 standardized features. Fit on ALL labelled
    Stage-2-eligible events (cultivation / cultivation_pasture /
    permanent_encroachment) at deployment time - this is the final-fit
    model that runs in production, distinct from the LOOCV folds used
    only to estimate accuracy (stage2_loocv.py, decisions.md 2026-09-29)."""

    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[LogisticRegression] = None
        self._n_train: int = 0

    @classmethod
    def fit_from_labelled_events(cls, rows: list[dict]) -> "Stage2Classifier":
        """
        rows: list of dicts, each with STAGE2_FEATURES plus 'known_class'.
        known_class in {'cultivation', 'cultivation_pasture'} -> label 0
        known_class == 'permanent_encroachment' -> label 1
        Any other known_class (natural, uncertain*, false_positive) is
        dropped - Stage 2 only ever sees Stage 1's "human" output, and
        `uncertain` rows have no resolved ground truth to train against.
        """
        eligible = [r for r in rows if r["known_class"] in (
            "cultivation", "cultivation_pasture", "permanent_encroachment"
        )]
        X = np.array([[r[f] for f in STAGE2_FEATURES] for r in eligible])
        y = np.array([1 if r["known_class"] == "permanent_encroachment" else 0 for r in eligible])

        clf = cls()
        clf._scaler = StandardScaler().fit(X)
        clf._model = LogisticRegression(max_iter=1000).fit(clf._scaler.transform(X), y)
        clf._n_train = len(eligible)
        return clf

    def classify(self, features: dict) -> Stage2Result:
        if self._model is None:
            raise RuntimeError("Stage2Classifier not fitted - call fit_from_labelled_events first")
        x = np.array([[features[f] for f in STAGE2_FEATURES]])
        x_scaled = self._scaler.transform(x)
        prob_permanent = float(self._model.predict_proba(x_scaled)[0][1])
        label = "permanent_encroachment" if prob_permanent >= 0.5 else "cultivation"
        note = (
            f"P(permanent_encroachment)={prob_permanent:.3f}, "
            f"trained on n={self._n_train} labelled events. "
            f"Does not attempt uncertain detection (decisions.md 2026-09-28) - "
            f"always returns a lean, never 'uncertain'."
        )
        return Stage2Result(
            label=label,
            probability_permanent=prob_permanent,
            confidence=_confidence_band(prob_permanent),
            evidence_note=note,
        )
