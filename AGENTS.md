# @wlearn/libsvm

LIBSVM v3.37 compiled to WebAssembly via Emscripten. Published as `@wlearn/libsvm` on npm.

## Architecture

```
LIBSVM v3.37 (C/C++)
       |
       |  csrc/wl_api.c (C wrapper: dense->sparse, buffer I/O, batch predict)
       |
       |  Emscripten (emcc, SINGLE_FILE=1)
       v
  wasm/svm.js (single file, embedded .wasm)
       |
       |  Direct wasm._function() calls
       v
  src/wasm.js       <- WASM loader (singleton, lazy init)
  src/model.js      <- SVMModel class (Estimator contract)
  src/index.js      <- Public API + convenience
```

## Type IDs

- `wlearn.libsvm.classifier@1` -- C_SVC, NU_SVC, ONE_CLASS
- `wlearn.libsvm.regressor@1` -- EPSILON_SVR, NU_SVR

## C wrapper design (`csrc/wl_api.c`)

Same pattern as @wlearn/liblinear:

### Dense-to-sparse conversion
- Pool allocation: `nrow * (ncol + 1)` svm_node pre-allocated
- Skip zeros, sentinel `{ index: -1, value: 0 }`
- No bias feature (LIBSVM handles bias internally)

### Buffer-based model I/O (MEMFS)
- `wl_svm_save_model`: MEMFS write + read back as buffer
- `wl_svm_load_model`: MEMFS write + load from virtual file
- Unique paths via atomic counter
- JS calls `wl_svm_free_buffer(ptr)` after copying

### Batch predict
- All predict functions loop rows in C
- Single scratch svm_node buffer reused across rows

### gamma default
- When gamma <= 0, defaults to 1/ncol (same as sklearn convention)

## Exported C API functions

```
wl_svm_get_last_error
wl_svm_train
wl_svm_predict
wl_svm_predict_probability
wl_svm_predict_values
wl_svm_save_model
wl_svm_load_model
wl_svm_free_model
wl_svm_free_buffer
wl_svm_get_nr_class
wl_svm_get_labels
wl_svm_get_sv_count
```

## Decision function output shapes

- Binary C_SVC/NU_SVC: `nrow` values (single margin)
- Multi-class (k classes): `nrow * k*(k-1)/2` pairwise margins
- ONE_CLASS: `nrow` values (positive = inlier, negative = outlier)
- SVR: `nrow` values

## Upstream tracking

- **Project**: LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- **Version**: 3.37
- **License**: Modified BSD
- **Modifications**: C wrapper only. No patches to upstream source.
- **What changed for WASM**: Dense-to-sparse conversion, MEMFS model I/O, batch prediction, single-threaded.
