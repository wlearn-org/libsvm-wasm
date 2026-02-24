export { loadSVM, getWasm } from './wasm.js'
export { SVMModel, SVMType, Kernel } from './model.js'

// Convenience: create, fit, return fitted model
export async function train(params, X, y) {
  const model = await (await import('./model.js')).SVMModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load model and predict, auto-disposes model
export async function predict(modelBuffer, X) {
  const model = await (await import('./model.js')).SVMModel.load(modelBuffer)
  const result = model.predict(X)
  model.dispose()
  return result
}
