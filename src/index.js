const { loadSVM, getWasm } = require('./wasm.js')
const { SVMModel, SVMType, Kernel } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await SVMModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await SVMModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadSVM, getWasm, SVMModel, SVMType, Kernel, train, predict }
