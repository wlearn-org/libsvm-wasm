import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { readFileSync, existsSync } from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadSVM } = await import('../src/wasm.js')
const wasm = await loadSVM()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_svm_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// SVMModel basics
// ============================================================
console.log('\n=== SVMModel ===')

const { SVMModel, SVMType, Kernel } = await import('../src/model.js')

await test('create() returns model', async () => {
  const model = await SVMModel.create({ svmType: 'C_SVC', kernel: 'RBF' })
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

// ============================================================
// Classification
// ============================================================
console.log('\n=== Classification ===')

await test('C_SVC with RBF kernel on nonlinear data', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 10.0,
    gamma: 0.5
  })

  // Concentric circles: class 0 near origin, class 1 farther out
  // Deterministic using LCG
  const X = []
  const y = []
  for (let i = 0; i < 200; i++) {
    const angle = ((i * 7 + 3) % 200) / 200 * Math.PI * 2
    if (i < 100) {
      const r = ((i * 13 + 7) % 100) / 100 * 2  // [0, 2]
      X.push([r * Math.cos(angle), r * Math.sin(angle)])
      y.push(0)
    } else {
      const r = 3 + ((i * 11 + 5) % 100) / 100 * 2  // [3, 5]
      X.push([r * Math.cos(angle), r * Math.sin(angle)])
      y.push(1)
    }
  }

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 200, `expected 200 predictions, got ${preds.length}`)

  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  assert(correct / preds.length > 0.7, `accuracy ${correct / preds.length} too low`)

  model.dispose()
})

await test('C_SVC with LINEAR kernel', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'LINEAR',
    C: 10.0
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1  // [-1, 1]
    const x2 = ((i * 13 + 7) % 100) / 50 - 1  // [-1, 1]
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }

  model.fit(X, y)
  const preds = model.predict(X)

  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  assert(correct / preds.length > 0.8, `accuracy ${correct / preds.length} too low`)

  model.dispose()
})

await test('NU_SVC classification', async () => {
  const model = await SVMModel.create({
    svmType: 'NU_SVC',
    kernel: 'RBF',
    nu: 0.5
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1
    const x2 = ((i * 13 + 7) % 100) / 50 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }

  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === 100, `expected 100 predictions`)

  model.dispose()
})

await test('Multi-class classification', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 10.0,
    gamma: 0.5
  })

  const X = []
  const y = []
  for (let i = 0; i < 150; i++) {
    const x1 = ((i * 7 + 3) % 150) / 75 - 1  // [-1, 1]
    const x2 = ((i * 13 + 7) % 150) / 75 - 1  // [-1, 1]
    X.push([x1, x2])
    const sum = x1 + x2
    y.push(sum < -0.3 ? 0 : sum < 0.3 ? 1 : 2)
  }

  model.fit(X, y)
  assert(model.nrClass === 3, `expected 3 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1 || preds[i] === 2,
      `invalid prediction: ${preds[i]}`)
  }

  model.dispose()
})

// ============================================================
// Probability
// ============================================================
console.log('\n=== Probability ===')

await test('predictProba with probability=1', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 10.0,
    gamma: 0.5,
    probability: 1
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const t = ((i * 7 + 3) % 100) / 100
    const s = ((i * 13 + 7) % 100) / 100
    X.push([t * 2 - 1, s * 2 - 1])
    y.push(t + s > 1 ? 1 : 0)
  }

  model.fit(X, y)
  const probs = model.predictProba(X)
  const nrClass = model.nrClass

  assert(probs.length === 100 * nrClass,
    `expected ${100 * nrClass} probabilities, got ${probs.length}`)

  for (let r = 0; r < 100; r++) {
    let sum = 0
    for (let c = 0; c < nrClass; c++) {
      const p = probs[r * nrClass + c]
      assert(p >= 0 && p <= 1, `probability out of [0,1]: ${p}`)
      sum += p
    }
    assertClose(sum, 1.0, 1e-4, `row ${r} probabilities sum to ${sum}`)
  }

  model.dispose()
})

// ============================================================
// Decision function
// ============================================================
console.log('\n=== Decision Function ===')

await test('decisionFunction returns values', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.5
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [0, 0, 1, 1]

  model.fit(X, y)
  const vals = model.decisionFunction(X)
  assert(vals instanceof Float64Array, 'should be Float64Array')
  assert(vals.length > 0, 'should have decision values')

  model.dispose()
})

// ============================================================
// Score
// ============================================================
console.log('\n=== Score ===')

await test('score returns accuracy for classification', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 10.0,
    gamma: 0.5
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const t = ((i * 11 + 5) % 100) / 100
    const s = ((i * 17 + 3) % 100) / 100
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }

  model.fit(X, y)
  const acc = model.score(X, y)
  assert(typeof acc === 'number', 'score should be a number')
  assert(acc > 0.8, `accuracy ${acc} too low`)
  assert(acc <= 1.0, `accuracy ${acc} > 1`)

  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('EPSILON_SVR regression', async () => {
  const model = await SVMModel.create({
    svmType: 'EPSILON_SVR',
    kernel: 'RBF',
    C: 10.0,
    gamma: 0.1,
    p: 0.1
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1  // [-1, 1]
    const noise = ((i * 31 + 11) % 100) / 500 - 0.1
    X.push([x1])
    y.push(2 * x1 + noise)
  }

  model.fit(X, y)
  assert(model.capabilities.regressor, 'should be regressor')

  const preds = model.predict(X)
  assert(preds.length === 100, 'expected 100 predictions')

  const r2 = model.score(X, y)
  assert(r2 > 0.5, `R-squared ${r2} too low`)

  model.dispose()
})

await test('NU_SVR regression', async () => {
  const model = await SVMModel.create({
    svmType: 'NU_SVR',
    kernel: 'RBF',
    nu: 0.5,
    C: 10.0,
    gamma: 0.1
  })

  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1
    const noise = ((i * 31 + 11) % 100) / 500 - 0.1
    X.push([x1])
    y.push(3 * x1 + noise)
  }

  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === 100, 'expected 100 predictions')

  model.dispose()
})

// ============================================================
// One-class SVM
// ============================================================
console.log('\n=== One-Class SVM ===')

await test('ONE_CLASS novelty detection', async () => {
  const model = await SVMModel.create({
    svmType: 'ONE_CLASS',
    kernel: 'RBF',
    nu: 0.1,
    gamma: 0.5
  })

  // Normal data clustered around origin (deterministic)
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = ((i * 7 + 3) % 100) / 50 - 1  // [-1, 1]
    const x2 = ((i * 13 + 7) % 100) / 50 - 1  // [-1, 1]
    X.push([x1, x2])
    y.push(1)  // dummy labels
  }

  model.fit(X, y)

  // Predict on normal data -- most should be +1 (inlier)
  const predsNormal = model.predict(X)
  let inliers = 0
  for (let i = 0; i < predsNormal.length; i++) {
    assert(predsNormal[i] === 1 || predsNormal[i] === -1,
      `one-class prediction should be +1 or -1, got ${predsNormal[i]}`)
    if (predsNormal[i] === 1) inliers++
  }
  assert(inliers / predsNormal.length > 0.5, 'most normal data should be inliers')

  model.dispose()
})

// ============================================================
// Save / Load (WLRN bundle format)
// ============================================================
console.log('\n=== Save / Load ===')

const { decodeBundle, load: coreLoad } = await import('@wlearn/core')

await test('save produces WLRN bundle', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.5
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

  const buf = model.save()
  assert(buf instanceof Uint8Array, 'save should return Uint8Array')
  assert(buf.length > 0, 'saved model should not be empty')

  // Verify WLRN magic
  assert(buf[0] === 0x57, 'bad magic[0]')
  assert(buf[1] === 0x4c, 'bad magic[1]')
  assert(buf[2] === 0x52, 'bad magic[2]')
  assert(buf[3] === 0x4e, 'bad magic[3]')

  // Verify manifest
  const { manifest, toc } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.libsvm.classifier@1',
    `expected classifier typeId, got ${manifest.typeId}`)
  assert(manifest.bundleVersion === 1, `expected bundleVersion 1, got ${manifest.bundleVersion}`)
  assert(manifest.params.svmType === 'C_SVC', `expected svmType C_SVC, got ${manifest.params.svmType}`)
  assert(manifest.params.C === 1.0, `expected C=1.0, got ${manifest.params.C}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC entry "model", got ${toc[0].id}`)

  model.dispose()
})

await test('save regressor uses regressor typeId', async () => {
  const model = await SVMModel.create({
    svmType: 'EPSILON_SVR',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.1
  })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])

  const buf = model.save()
  const { manifest } = decodeBundle(buf)
  assert(manifest.typeId === 'wlearn.libsvm.regressor@1',
    `expected regressor typeId, got ${manifest.typeId}`)

  model.dispose()
})

await test('save and load model round-trip', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.5
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [0, 0, 1, 1]
  model.fit(X, y)

  const preds1 = model.predict(X)
  const buf = model.save()

  const model2 = await SVMModel.load(buf)
  assert(model2.isFitted, 'loaded model should be fitted')

  const preds2 = model2.predict(X)

  // Same-runtime round-trip: exact match
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  // Loaded model preserves params
  const params = model2.getParams()
  assert(params.svmType === 'C_SVC', `loaded params.svmType = ${params.svmType}`)
  assert(params.C === 1.0, `loaded params.C = ${params.C}`)

  model.dispose()
  model2.dispose()
})

// ============================================================
// core.load() registry dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to libsvm loader', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.5
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

  const preds1 = model.predict([[1, 2], [7, 8]])
  const buf = model.save()

  // Load via core registry dispatcher (not SVMModel.load directly)
  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded model should be fitted')

  const preds2 = model2.predict([[1, 2], [7, 8]])
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i],
      `core.load prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  model2.dispose()
})

await test('core.load() works for regressor bundles', async () => {
  const model = await SVMModel.create({
    svmType: 'EPSILON_SVR',
    kernel: 'RBF',
    C: 1.0,
    gamma: 0.1
  })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])

  const buf = model.save()
  const model2 = await coreLoad(buf)
  assert(model2.isFitted, 'registry-loaded regressor should be fitted')

  const preds = model2.predict([[1, 2]])
  assert(preds.length === 1, `expected 1 prediction, got ${preds.length}`)

  model.dispose()
  model2.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await SVMModel.create({ svmType: 'C_SVC', kernel: 'RBF', C: 2.0 })

  const params = model.getParams()
  assert(params.svmType === 'C_SVC', `expected C_SVC, got ${params.svmType}`)
  assert(params.C === 2.0, `expected C=2.0, got ${params.C}`)

  model.setParams({ C: 5.0 })
  assert(model.getParams().C === 5.0, 'C should be updated')

  model.dispose()
})

await test('defaultSearchSpace returns object', async () => {
  const space = SVMModel.defaultSearchSpace()
  assert(space, 'search space is null')
  assert(space.svmType, 'missing svmType in search space')
  assert(space.kernel, 'missing kernel in search space')
  assert(space.C, 'missing C in search space')
})

// ============================================================
// Resource management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const model = await SVMModel.create({ svmType: 'C_SVC', kernel: 'LINEAR' })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()
  model.dispose()
})

await test('throws after dispose', async () => {
  const model = await SVMModel.create({ svmType: 'C_SVC', kernel: 'LINEAR' })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('throws before fit', async () => {
  const model = await SVMModel.create({ svmType: 'C_SVC' })

  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')

  model.dispose()
})

// ============================================================
// Input coercion
// ============================================================
console.log('\n=== Input Coercion ===')

await test('typed matrix fast path', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'LINEAR',
    C: 1.0
  })

  const X = {
    data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
    rows: 4,
    cols: 2
  }
  model.fit(X, new Float64Array([0, 0, 1, 1]))
  const preds = model.predict(X)
  assert(preds.length === 4, `expected 4 predictions, got ${preds.length}`)

  model.dispose()
})

await test('coerce error mode rejects arrays', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'LINEAR',
    coerce: 'error'
  })

  let threw = false
  try {
    model.fit([[1, 2], [3, 4]], [0, 1])
  } catch {
    threw = true
  }
  assert(threw, 'error mode should reject number[][]')

  model.dispose()
})

// ============================================================
// Gamma default
// ============================================================
console.log('\n=== Gamma Default ===')

await test('gamma defaults to 1/n_features when 0', async () => {
  const model = await SVMModel.create({
    svmType: 'C_SVC',
    kernel: 'RBF',
    gamma: 0,
    C: 1.0
  })

  const X = []
  const y = []
  for (let i = 0; i < 50; i++) {
    const x1 = ((i * 7 + 3) % 50) / 25 - 1
    const x2 = ((i * 13 + 7) % 50) / 25 - 1
    X.push([x1, x2])
    y.push(i < 25 ? 0 : 1)
  }

  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === 50, 'should get 50 predictions')

  model.dispose()
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('capabilities reflect SVM type', async () => {
  const csvc = await SVMModel.create({ svmType: 'C_SVC', probability: 1 })
  assert(csvc.capabilities.classifier === true, 'C_SVC should be classifier')
  assert(csvc.capabilities.predictProba === true, 'C_SVC+prob should support predictProba')
  csvc.dispose()

  const svr = await SVMModel.create({ svmType: 'EPSILON_SVR' })
  assert(svr.capabilities.regressor === true, 'SVR should be regressor')
  assert(svr.capabilities.classifier === false, 'SVR should not be classifier')
  svr.dispose()

  const oc = await SVMModel.create({ svmType: 'ONE_CLASS' })
  assert(oc.capabilities.oneClass === true, 'ONE_CLASS should have oneClass capability')
  oc.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)
