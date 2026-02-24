# Changelog

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
