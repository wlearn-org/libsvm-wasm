// WASM loader -- loads the LIBSVM WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

export async function loadSVM(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const { default: createSVM } = await import('../wasm/svm.js')
    wasmModule = await createSVM(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadSVM() first')
  return wasmModule
}
