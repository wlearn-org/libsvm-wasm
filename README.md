# @wlearn/libsvm

LIBSVM v3.37 compiled to WebAssembly. Kernel SVM classification, regression, and novelty detection in browsers and Node.js.

Based on [LIBSVM v3.37](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (BSD-3-Clause). Zero dependencies. CommonJS.

## Install

```bash
npm install @wlearn/libsvm
```

## Quick start

```js
const { SVMModel } = require('@wlearn/libsvm')

const model = await SVMModel.create({
  svmType: 'C_SVC',
  kernel: 'RBF',
  C: 1.0,
  gamma: 0.5
})

// Train -- accepts number[][] or { data: Float64Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])  // Float64Array

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
const buf = model.save()  // Uint8Array
const model2 = await SVMModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## Typed matrix input (fast path)

```js
const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
  rows: 4,
  cols: 2
}
model.fit(X, new Float64Array([0, 0, 1, 1]))
```

## Input coercion policy

```js
const model = await SVMModel.create({ coerce: 'auto' })   // convert silently (default)
const model = await SVMModel.create({ coerce: 'warn' })    // warn once per instance
const model = await SVMModel.create({ coerce: 'error' })   // throw on non-typed input
```

## API

### `SVMModel.create(params?)`

Async factory. Loads WASM module, returns ready-to-use model.

Parameters:
- `svmType` -- `'C_SVC'` | `'NU_SVC'` | `'ONE_CLASS'` | `'EPSILON_SVR'` | `'NU_SVR'` (default: `'C_SVC'`)
- `kernel` -- `'LINEAR'` | `'POLY'` | `'RBF'` | `'SIGMOID'` (default: `'RBF'`)
- `C` -- regularization (default: `1.0`)
- `gamma` -- kernel coefficient, <= 0 means 1/n_features (default: `0`)
- `degree` -- polynomial degree (default: `3`)
- `coef0` -- independent term in kernel (default: `0`)
- `nu` -- for NU_SVC/NU_SVR/ONE_CLASS (default: `0.5`)
- `eps` -- stopping tolerance (default: `0.001`)
- `p` -- epsilon in SVR loss (default: `0.1`)
- `shrinking` -- use shrinking heuristic (default: `1`)
- `probability` -- enable probability estimates (default: `0`)
- `cacheSize` -- kernel cache in MB (default: `100`)
- `coerce` -- `'auto'` | `'warn'` | `'error'` (default: `'auto'`)

### `model.fit(X, y)`

Train on data. Returns `this`.

### `model.predict(X)`

Returns `Float64Array` of predicted labels.

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major probabilities).
Requires `probability: 1` in constructor params.

### `model.decisionFunction(X)`

Returns `Float64Array` of decision values.
- Binary classification: `nrow` values
- Multi-class: `nrow * nr_class*(nr_class-1)/2` pairwise margins

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()` / `SVMModel.load(buffer)`

Save to / load from `Uint8Array` (native LIBSVM format).

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters.

### `SVMModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## SVM types

| Name | Code | Task |
|------|------|------|
| C_SVC | 0 | C-support vector classification |
| NU_SVC | 1 | nu-support vector classification |
| ONE_CLASS | 2 | One-class SVM (novelty detection) |
| EPSILON_SVR | 3 | epsilon-support vector regression |
| NU_SVR | 4 | nu-support vector regression |

## Kernels

| Name | Code | Formula |
|------|------|---------|
| LINEAR | 0 | u'*v |
| POLY | 1 | (gamma*u'*v + coef0)^degree |
| RBF | 2 | exp(-gamma*\|u-v\|^2) |
| SIGMOID | 3 | tanh(gamma*u'*v + coef0) |

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every model when done.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/libsvm-wasm
cd libsvm-wasm
npm run build
npm test
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## License

BSD-3-Clause (same as upstream LIBSVM)
