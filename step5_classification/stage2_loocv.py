import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Stage 2 population: 19 events, known_class in {cultivation, cultivation_pasture, permanent_encroachment}
# label: 0 = cultivation-side (cultivation + cultivation_pasture), 1 = permanent_encroachment
data = [
    ("REF_004", "cultivation", 0.0008664585828426792, 0.13175035824819645, 9),
    ("REF_055", "cultivation", 0.00043253262323381987, 0.140519375490064, 8),
    ("REF_058", "cultivation", -0.00019082395072820963, 0.12385635025391804, 7),
    ("REF_053", "permanent_encroachment", 0.0003987669022357581, 0.08972200213076971, 9),
    ("REF_013", "permanent_encroachment", -0.000987215219478381, 0.1379522556891605, 7),
    ("REF_008", "cultivation_pasture", 0.0008582432748012702, 0.16611115163134357, 6),
    ("REF_021", "cultivation", 0.0005286965397977237, 0.1347819260942866, 6),
    ("REF_016", "cultivation", 0.0009763577348490727, 0.20202602548085463, 6),
    ("REF_044", "permanent_encroachment", 0.0005806974776395681, 0.09005486329872656, 4),
    ("REF_034", "cultivation", 0.00022437504533108122, 0.15708195942046538, 4),
    ("REF_049", "permanent_encroachment", -0.0005780649811023556, 0.08984477795359654, 3),
    ("REF_052", "cultivation", 0.0015335559984989754, 0.24636638381853332, 5),
    ("REF_014", "cultivation", 0.0009850863000567972, 0.14197498548442208, 6),
    ("REF_023", "cultivation", 0.0009109343132845324, 0.1995204627509644, 9),
    ("REF_012", "permanent_encroachment", 0.0004181490219628562, 0.08098715250501912, 5),
    ("REF_007", "cultivation", 0.0009629128054821398, 0.18894753392243696, 3),
    ("REF_033", "cultivation", 0.0005179603029270706, 0.1401467010141624, 6),
    ("REF_031", "cultivation", 0.0007539543366889299, 0.14565204274617655, 5),
    ("REF_025", "permanent_encroachment", 0.0007534907341969593, 0.20245170969366513, 1),
]

ids = [d[0] for d in data]
labels_raw = [d[1] for d in data]
y = np.array([1 if l == "permanent_encroachment" else 0 for l in labels_raw])
X = np.array([[d[2], d[3], d[4]] for d in data])

print(f"n = {len(y)}, permanent_encroachment = {y.sum()}, cultivation-side = {len(y)-y.sum()}")
print(f"Majority class baseline accuracy = {max(y.sum(), len(y)-y.sum())/len(y):.3f}\n")

loo = LeaveOneOut()

def run_loocv(model_fn, name, scale=False):
    preds = []
    probs = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model = model_fn()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        preds.append(pred)
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(X_test)[0][1])
        else:
            probs.append(None)
    preds = np.array(preds)
    acc = (preds == y).mean()
    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    print(f"--- {name} ---")
    print(f"LOOCV accuracy: {acc:.3f} ({(preds==y).sum()}/{len(y)})")
    print(f"Confusion (pos=permanent_encroachment): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f}")
    print("Per-event: id, true, pred, prob(perm_encroach)")
    for i, (id_, t, p, pr) in enumerate(zip(ids, y, preds, probs)):
        mark = "  " if t == p else "**MISS**"
        prstr = f"{pr:.3f}" if pr is not None else "n/a"
        print(f"  {id_:10s} true={t} pred={p} prob={prstr} {mark}")
    print()
    return preds, probs

run_loocv(lambda: LogisticRegression(max_iter=1000), "Logistic Regression (standardized features)", scale=True)
run_loocv(lambda: DecisionTreeClassifier(max_depth=1, random_state=0), "Decision Stump (max_depth=1)")
run_loocv(lambda: DecisionTreeClassifier(max_depth=2, random_state=0), "Decision Tree (max_depth=2)")

# Fit full-data tree just to see what split it would use (NOT for evaluation, illustrative only)
full_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
print("--- Full-data tree structure (illustrative only, NOT an evaluation result) ---")
print(export_text(full_tree, feature_names=["trend_nbr_per_day", "wobble_std", "cycle_count_provisional"]))

full_lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X), y)
print("--- Full-data logistic regression coefficients (illustrative only, NOT an evaluation result) ---")
for name, coef in zip(["trend_nbr_per_day", "wobble_std", "cycle_count_provisional"], full_lr.coef_[0]):
    print(f"  {name}: {coef:.3f}")
