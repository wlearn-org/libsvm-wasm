#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GLUE="${1:-${PROJECT_DIR}/wasm/svm.cjs}"

if [ ! -f "$GLUE" ]; then
  echo "ERROR: glue file not found: $GLUE"
  exit 1
fi

REQUIRED_SYMBOLS=(
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
)

missing=0
for fn in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "_${fn}" "$GLUE"; then
    echo "MISSING: ${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} symbol(s) missing"
  exit 1
fi

echo "All ${#REQUIRED_SYMBOLS[@]} exports verified"
