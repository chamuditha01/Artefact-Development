"""
run_e2.py

E2: per-stage AND end-to-end classification accuracy, on the frozen REF
evaluation set (never used for calibration - decisions.md 2026-09-19).

Population: the 20 REF events with a resolved (non-uncertain, non-logging-
pooled) 3-class label: 13 cultivation-side (12 cultivation + 1
cultivation_pasture), 6 permanent_encroachment, 1 natural.

Method:
  Stage 1 - hand-set thresholds (classifiers.py), NOT retrained per event
  (it is a fixed rule, not a fitted model). Applied directly to all 20.

  Stage 2 - logistic regression, LOOCV over the 19 Stage-1-"human" events
  (same method as stage2_loocv.py / decisions.md 2026-09-29): for each
  event, fit on the other 18, predict the held-out one.

  End-to-end - for each of the 20 events: run Stage 1. If it says
  candidate_natural, the final predicted class is "natural" (no Stage 2
  call - matches how the real pipeline would behave: Stage 2 only ever
  sees Stage 1's "human" output). If candidate_human, the final predicted
  class is that event's Stage 2 LOOCV prediction. A Stage 1 mistake is
  therefore unrecoverable in this count, exactly as decisions.md's
  cascade-error limitation describes.

Caveat stated in the printed output, not just here: natural has n=1 in
this set. Any accuracy figure that includes it is not a reliable rate for
that class - SKILL.md's own guardrail.

RESULT (run 2026-10-01, REF set, n=20):
  Stage 1 alone        : 19/20 = 0.950  (n_natural=1, not a reliable rate)
  Stage 2 alone (LOOCV): 15/19 = 0.789  (human-subset only, n=19)
  End-to-end (cascade) : 15/20 = 0.750  (n=20, all resolved events)
"""

import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step5_classification'))
from classifiers import stage1_classify, STAGE2_FEATURES

RESOLVED_CLASSES = {"cultivation", "cultivation_pasture", "permanent_encroachment", "natural"}


def true_class_3way(known_class: str) -> str:
    if known_class in ("cultivation", "cultivation_pasture"):
        return "cultivation"
    return known_class  # permanent_encroachment | natural


def load_rows(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            row = dict(r)
            for k in ["shape_compactness", "slope_deg", "dist_to_prior_loss_m",
                      "trend_nbr_per_day", "wobble_std", "cycle_count_provisional"]:
                row[k] = float(row[k])
            rows.append(row)
        return rows


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    all_rows = load_rows(os.path.join(data_dir, "features_REF_merged.csv"))
    resolved = [r for r in all_rows if r["known_class"] in RESOLVED_CLASSES]
    print(f"Resolved REF population: n={len(resolved)}")
    counts = {}
    for r in resolved:
        c = true_class_3way(r["known_class"])
        counts[c] = counts.get(c, 0) + 1
    print(f"  by 3-class truth: {counts}\n")

    # ---------- Stage 1: run directly (fixed rule, not fitted) ----------
    print("=" * 70)
    print("STAGE 1 (human vs natural) - fixed thresholds, all 20 events")
    print("=" * 70)
    stage1_correct = 0
    stage1_records = []
    for r in resolved:
        true3 = true_class_3way(r["known_class"])
        true_stage1 = "candidate_natural" if true3 == "natural" else "candidate_human"
        res = stage1_classify(r)
        correct = (res.label == true_stage1)
        stage1_correct += correct
        stage1_records.append((r["event_id"], true3, true_stage1, res.label, correct))
        mark = "" if correct else "  **MISS**"
        print(f"  {r['event_id']:10s} true={true3:22s} stage1_true={true_stage1:17s} "
              f"stage1_pred={res.label:17s}{mark}")
    stage1_acc = stage1_correct / len(resolved)
    print(f"\nStage 1 accuracy: {stage1_correct}/{len(resolved)} = {stage1_acc:.3f}")
    n_natural = sum(1 for r in resolved if true_class_3way(r["known_class"]) == "natural")
    print(f"CAVEAT: natural class n={n_natural} in this set. Not a reliable rate for "
          f"that class on its own (SKILL.md guardrail) - reported here only as part of "
          f"the end-to-end figure below, never standalone.\n")

    # ---------- Stage 2: LOOCV over the 19 "human" events ----------
    print("=" * 70)
    print("STAGE 2 (cultivation vs permanent_encroachment) - LOOCV, n=19 human events")
    print("=" * 70)
    human = [r for r in resolved if true_class_3way(r["known_class"]) != "natural"]
    ids = [r["event_id"] for r in human]
    y = np.array([1 if r["known_class"] == "permanent_encroachment" else 0 for r in human])
    X = np.array([[r[f] for f in STAGE2_FEATURES] for r in human])

    loo = LeaveOneOut()
    stage2_preds = {}  # event_id -> predicted 3-class label, from LOOCV
    preds_arr = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
        pred = model.predict(scaler.transform(X_test))[0]
        i = test_idx[0]
        preds_arr[i] = pred
        stage2_preds[ids[i]] = "permanent_encroachment" if pred == 1 else "cultivation"

    full_scaler = StandardScaler().fit(X)
    full_model = LogisticRegression(max_iter=1000).fit(full_scaler.transform(X), y)

    def stage2_predict_fulldata(features: dict) -> str:
        x = np.array([[features[f] for f in STAGE2_FEATURES]])
        p = full_model.predict(full_scaler.transform(x))[0]
        return "permanent_encroachment" if p == 1 else "cultivation"

    stage2_acc = (preds_arr == y).mean()
    for eid, t, p in zip(ids, y, preds_arr):
        mark = "" if t == p else "  **MISS**"
        tcls = "permanent_encroachment" if t == 1 else "cultivation"
        pcls = "permanent_encroachment" if p == 1 else "cultivation"
        print(f"  {eid:10s} true={tcls:22s} pred={pcls:22s}{mark}")
    print(f"\nStage 2 accuracy (standalone LOOCV, n=19): "
          f"{(preds_arr == y).sum()}/{len(y)} = {stage2_acc:.3f}\n")

    # ---------- End-to-end: Stage 1 routes, Stage 2 (LOOCV) decides within "human" ----------
    print("=" * 70)
    print("END-TO-END (Stage 1 -> Stage 2 cascade) - n=20, all resolved events")
    print("=" * 70)
    e2e_correct = 0
    for r in resolved:
        eid = r["event_id"]
        true3 = true_class_3way(r["known_class"])
        s1 = stage1_classify(r)
        if s1.label == "candidate_natural":
            final_pred = "natural"
        elif eid in stage2_preds:
            final_pred = stage2_preds[eid]
        else:
            final_pred = stage2_predict_fulldata(r)
            eid = eid + " (misrouted-by-S1, full-fit S2 used)"
        correct = (final_pred == true3)
        e2e_correct += correct
        mark = "" if correct else "  **MISS**"
        print(f"  {eid:10s} true={true3:22s} stage1={s1.label:17s} final_pred={final_pred:22s}{mark}")

    e2e_acc = e2e_correct / len(resolved)
    print(f"\nEnd-to-end accuracy: {e2e_correct}/{len(resolved)} = {e2e_acc:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Stage 1 alone        : {stage1_correct}/{len(resolved)} = {stage1_acc:.3f}  (n_natural={n_natural}, caveat above)")
    print(f"  Stage 2 alone (LOOCV): {(preds_arr==y).sum()}/{len(y)} = {stage2_acc:.3f}  (human-subset only, n=19)")
    print(f"  End-to-end (cascade) : {e2e_correct}/{len(resolved)} = {e2e_acc:.3f}  (n=20, all resolved events)")
    cost = (preds_arr == y).sum() - e2e_correct
    print(f"\n  Cascade cost: end-to-end is {cost} event(s) worse than Stage 2 alone,"
          f" out of {len(resolved)-len(y)} event(s) not covered by the Stage-2-only figure.")


if __name__ == "__main__":
    main()
