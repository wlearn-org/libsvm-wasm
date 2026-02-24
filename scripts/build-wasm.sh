#!/bin/bash
set -euo pipefail

# Build LIBSVM v3.37 as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/libsvm"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/svm.h" ]; then
  echo "ERROR: LIBSVM upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null 2>&1; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "$UPSTREAM_DIR" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='["_wl_svm_get_last_error","_wl_svm_train","_wl_svm_predict","_wl_svm_predict_probability","_wl_svm_predict_values","_wl_svm_save_model","_wl_svm_load_model","_wl_svm_free_model","_wl_svm_free_buffer","_wl_svm_get_nr_class","_wl_svm_get_labels","_wl_svm_get_sv_count","_malloc","_free"]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","UTF8ToString"]'

emcc \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  "${UPSTREAM_DIR}/svm.cpp" \
  -I "${UPSTREAM_DIR}" \
  -o "${OUTPUT_DIR}/svm.js" \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createSVM \
  -s FORCE_FILESYSTEM=1 \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: libsvm v3.37
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/svm.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
