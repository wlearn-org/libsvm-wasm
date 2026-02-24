"""
Generate test fixtures for @wlearn/libsvm cross-runtime parity tests.

Requires: scikit-learn (which wraps LIBSVM internally)

Usage:
    python test/fixtures/generate.py
"""

import json
import numpy as np
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
np.random.seed(42)


def save_fixture(name, X, y, predictions, params):
    data = {
        'X': X.tolist(),
        'y': y.tolist(),
        'predictions': predictions.tolist(),
        'params': params
    }
    with open(FIXTURES_DIR / f'{name}.data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  Saved {name}.data.json ({len(X)} samples, {X.shape[1]} features)')


# --- RBF classification ---
print('RBF classification (C_SVC + RBF):')
# Concentric circles
n = 100
angles = np.random.rand(n) * 2 * np.pi
r_inner = np.random.rand(n // 2) * 2
r_outer = 3 + np.random.rand(n // 2) * 2
X_inner = np.column_stack([r_inner * np.cos(angles[:n//2]), r_inner * np.sin(angles[:n//2])])
X_outer = np.column_stack([r_outer * np.cos(angles[n//2:]), r_outer * np.sin(angles[n//2:])])
X = np.vstack([X_inner, X_outer]).astype(np.float64)
y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float64)

clf = SVC(kernel='rbf', C=10.0, gamma=0.5)
clf.fit(X, y)
preds = clf.predict(X)

save_fixture('rbf_classification', X, y, preds, {
    'svmType': 'C_SVC',
    'kernel': 'RBF',
    'C': 10.0,
    'gamma': 0.5,
})

# --- Linear classification ---
print('Linear classification (C_SVC + LINEAR):')
X_lin = np.random.randn(100, 2).astype(np.float64)
y_lin = (X_lin[:, 0] + X_lin[:, 1] > 0).astype(np.float64)

clf_lin = SVC(kernel='linear', C=1.0)
clf_lin.fit(X_lin, y_lin)
preds_lin = clf_lin.predict(X_lin)

save_fixture('linear_classification', X_lin, y_lin, preds_lin, {
    'svmType': 'C_SVC',
    'kernel': 'LINEAR',
    'C': 1.0,
})

# --- Regression (SVR) ---
print('Regression (EPSILON_SVR + RBF):')
X_reg = np.random.randn(100, 1).astype(np.float64) * 5
y_reg = (np.sin(X_reg[:, 0]) + np.random.randn(100) * 0.1).astype(np.float64)

reg = SVR(kernel='rbf', C=10.0, gamma=0.1, epsilon=0.1)
reg.fit(X_reg, y_reg)
preds_reg = reg.predict(X_reg)

save_fixture('regression', X_reg, y_reg, preds_reg, {
    'svmType': 'EPSILON_SVR',
    'kernel': 'RBF',
    'C': 10.0,
    'gamma': 0.1,
    'p': 0.1,
})

# --- One-class SVM ---
print('One-class SVM:')
X_oc = np.random.randn(100, 2).astype(np.float64)
y_oc = np.ones(100, dtype=np.float64)  # dummy

oc = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.5)
oc.fit(X_oc)
preds_oc = oc.predict(X_oc)

save_fixture('one_class', X_oc, y_oc, preds_oc.astype(np.float64), {
    'svmType': 'ONE_CLASS',
    'kernel': 'RBF',
    'nu': 0.1,
    'gamma': 0.5,
})

print('Done.')
