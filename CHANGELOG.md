# Changelog

## 0.2.0 (unreleased)

- Wrap SVMModel with `createModelClass` for unified task detection
- Add `task` parameter: `'classification'` or `'regression'`, auto-detected from labels if omitted

## 0.1.0 (unreleased)

- Initial release
- LIBSVM v3.37 compiled to WASM via Emscripten
- Unified sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `dispose()`
- Kernel SVM: C-SVC, nu-SVC, one-class SVM, epsilon-SVR, nu-SVR
- Kernels: linear, polynomial, RBF, sigmoid
- Buffer-based model I/O (no filesystem dependency)
- Accepts both typed matrices and number[][] with configurable coercion
- `predictProba()` for probability estimates
- `decisionFunction()` for decision values
- `getParams()`/`setParams()` for AutoML integration
- `defaultSearchSpace()` for hyperparameter search
- `FinalizationRegistry` safety net for leak detection
- BSD-3-Clause license (same as upstream LIBSVM)
