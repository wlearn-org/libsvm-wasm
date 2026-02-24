import { getWasm, loadSVM } from './wasm.js'

// FinalizationRegistry safety net
const registry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/libsvm: Model was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

// --- SVM type and kernel constants ---

export const SVMType = {
  C_SVC: 0,
  NU_SVC: 1,
  ONE_CLASS: 2,
  EPSILON_SVR: 3,
  NU_SVR: 4
}

export const Kernel = {
  LINEAR: 0,
  POLY: 1,
  RBF: 2,
  SIGMOID: 3
}

const SVR_TYPES = new Set([SVMType.EPSILON_SVR, SVMType.NU_SVR])

// --- Input normalization ---

function resolveSVMType(s) {
  if (typeof s === 'number') return s
  if (typeof s === 'string' && s in SVMType) return SVMType[s]
  return SVMType.C_SVC
}

function resolveKernel(k) {
  if (typeof k === 'number') return k
  if (typeof k === 'string' && k in Kernel) return Kernel[k]
  return Kernel.RBF
}

function normalizeX(X, coerce) {
  // Fast path: typed matrix
  if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
    const { data, rows, cols } = X
    if (!(data instanceof Float64Array)) {
      if (coerce === 'error') throw new Error('Expected Float64Array in typed matrix')
      return { data: new Float64Array(data), rows, cols }
    }
    return { data, rows, cols }
  }

  // Slow path: number[][]
  if (Array.isArray(X) && Array.isArray(X[0])) {
    if (coerce === 'error') {
      throw new Error('Input coercion disabled (coerce: "error"). Pass { data: Float64Array, rows, cols } instead of number[][].')
    }
    const rows = X.length
    const cols = X[0].length
    const data = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = X[i][j]
      }
    }
    if (coerce === 'warn') {
      const bytes = data.byteLength
      console.warn(`@wlearn/libsvm: Converted number[][] to Float64Array (copied ${(bytes / 1024).toFixed(1)} KB, shape ${rows}x${cols}). For performance, pass { data, rows, cols }.`)
    }
    return { data, rows, cols }
  }

  throw new Error('X must be number[][] or { data: Float64Array, rows, cols }')
}

function normalizeY(y) {
  if (y instanceof Float64Array) return y
  return new Float64Array(y)
}

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_svm_get_last_error', 'string', [], [])
}

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- SVMModel ---

export class SVMModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #coerce = 'auto'
  #warned = false
  #fitted = false
  #ncol = 0  // track feature count for gamma default

  constructor(handle, params, coerce) {
    if (handle === LOAD_SENTINEL) {
      this.#handle = params
      this.#params = coerce || {}
      this.#coerce = this.#params.coerce || 'auto'
      this.#fitted = true
    } else {
      this.#handle = null
      this.#params = handle || {}
      this.#coerce = this.#params.coerce || 'auto'
    }

    this.#freed = false
    if (this.#handle) {
      this.#ptrRef = [this.#handle]
      if (registry) {
        registry.register(this, {
          ptr: this.#ptrRef,
          freeFn: (h) => getWasm()._wl_svm_free_model(h)
        }, this)
      }
    }
  }

  static async create(params = {}) {
    await loadSVM()
    return new SVMModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    const wasm = getWasm()

    // Dispose previous model if refitting
    if (this.#handle) {
      wasm._wl_svm_free_model(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (registry) registry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yData = normalizeY(y)
    this.#ncol = cols

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    // Allocate on WASM heap
    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const svmType = resolveSVMType(this.#params.svmType)
    const kernel = resolveKernel(this.#params.kernel)
    const degree = this.#params.degree ?? 3
    const gamma = this.#params.gamma ?? 0  // 0 means 1/n_features in C wrapper
    const coef0 = this.#params.coef0 ?? 0
    const C = this.#params.C ?? 1.0
    const nu = this.#params.nu ?? 0.5
    const eps = this.#params.eps ?? 0.001
    const p = this.#params.p ?? 0.1
    const shrinking = this.#params.shrinking ?? 1
    const probability = this.#params.probability ?? 0
    const cacheSize = this.#params.cacheSize ?? 100

    const modelPtr = wasm._wl_svm_train(
      xPtr, rows, cols, yPtr,
      svmType, kernel, degree, gamma, coef0,
      C, nu, eps, p,
      shrinking, probability, cacheSize
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true

    this.#ptrRef = [this.#handle]
    if (registry) {
      registry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_svm_free_model(h)
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_svm_predict(this.#handle, xPtr, rows, cols, outPtr)

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    for (let i = 0; i < rows; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nrClass = this.nrClass

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * nrClass * 8)

    const ret = wasm._wl_svm_predict_probability(this.#handle, xPtr, rows, cols, outPtr)

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    const total = rows * nrClass
    const result = new Float64Array(total)
    for (let i = 0; i < total; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  decisionFunction(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const dimPtr = wasm._malloc(4)
    const nrClass = this.nrClass
    const maxDim = nrClass * (nrClass - 1) / 2 || 1
    const outPtr = wasm._malloc(rows * maxDim * 8)

    const ret = wasm._wl_svm_predict_values(
      this.#handle, xPtr, rows, cols, outPtr, dimPtr
    )

    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      wasm._free(dimPtr)
      throw new Error(`decisionFunction failed: ${getLastError()}`)
    }

    const dim = wasm.getValue(dimPtr, 'i32')
    const total = rows * dim
    const result = new Float64Array(total)
    for (let i = 0; i < total; i++) {
      result[i] = wasm.HEAPF64[outPtr / 8 + i]
    }

    wasm._free(xPtr)
    wasm._free(outPtr)
    wasm._free(dimPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#isRegressor()) {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    } else {
      // Accuracy
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
      }
      return correct / preds.length
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_svm_save_model(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_svm_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }

  static async load(buffer) {
    await loadSVM()
    const wasm = getWasm()

    const buf = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
    const bufPtr = wasm._malloc(buf.length)
    wasm.HEAPU8.set(buf, bufPtr)

    const modelPtr = wasm._wl_svm_load_model(bufPtr, buf.length)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    return new SVMModel(LOAD_SENTINEL, modelPtr, {})
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_svm_free_model(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (registry) registry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    if ('coerce' in p) this.#coerce = p.coerce
    return this
  }

  static defaultSearchSpace() {
    return {
      svmType: { type: 'categorical', values: ['C_SVC', 'NU_SVC'] },
      kernel: { type: 'categorical', values: ['RBF', 'LINEAR', 'POLY'] },
      C: { type: 'log_uniform', low: 1e-3, high: 1e3 },
      gamma: { type: 'log_uniform', low: 1e-5, high: 1e1 },
      degree: { type: 'int_uniform', low: 2, high: 5, condition: { kernel: 'POLY' } },
      nu: { type: 'uniform', low: 0.01, high: 0.99, condition: { svmType: 'NU_SVC' } }
    }
  }

  // --- Inspection ---

  get nrClass() {
    this.#ensureFitted()
    return getWasm()._wl_svm_get_nr_class(this.#handle)
  }

  get svCount() {
    this.#ensureFitted()
    return getWasm()._wl_svm_get_sv_count(this.#handle)
  }

  get classes() {
    this.#ensureFitted()
    const wasm = getWasm()
    const n = this.nrClass
    const outPtr = wasm._malloc(n * 4)
    wasm._wl_svm_get_labels(this.#handle, outPtr)
    const result = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      result[i] = wasm.getValue(outPtr + i * 4, 'i32')
    }
    wasm._free(outPtr)
    return result
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const svmType = resolveSVMType(this.#params.svmType)
    const isRegressor = SVR_TYPES.has(svmType)
    const probability = this.#params.probability ?? 0
    return {
      classifier: !isRegressor,
      regressor: isRegressor,
      predictProba: !isRegressor && probability === 1,
      decisionFunction: true,
      oneClass: svmType === SVMType.ONE_CLASS,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get probaDim() {
    return this.isFitted ? this.nrClass : 0
  }

  get decisionDim() {
    if (!this.isFitted) return 0
    const svmType = resolveSVMType(this.#params.svmType)
    if (svmType === SVMType.ONE_CLASS || SVR_TYPES.has(svmType)) return 1
    const n = this.nrClass
    return n === 2 ? 1 : n * (n - 1) / 2
  }

  // --- Private helpers ---

  #normalizeX(X) {
    const coerce = this.#warned ? 'auto' : this.#coerce
    const result = normalizeX(X, coerce)
    if (this.#coerce === 'warn' && !this.#warned && Array.isArray(X)) {
      this.#warned = true
    }
    return result
  }

  #ensureFitted() {
    if (this.#freed) throw new Error('Model already disposed')
    if (!this.#fitted) throw new Error('Model not fitted -- call fit() first')
  }

  #isRegressor() {
    const svmType = resolveSVMType(this.#params.svmType)
    return SVR_TYPES.has(svmType)
  }
}
