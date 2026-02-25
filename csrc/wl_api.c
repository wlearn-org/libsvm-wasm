/*
 * wl_api.c -- C wrapper for LIBSVM v3.37
 *
 * Bridges dense JS float64 arrays to LIBSVM's sparse svm_node format.
 * Provides buffer-based model I/O via Emscripten MEMFS.
 * Batch prediction (loop in C, not one JS->WASM call per row).
 *
 * Compile with: emcc csrc/wl_api.c upstream/libsvm/svm.cpp ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- error handling ---------- */

static char last_error[512] = "";

static void set_error(const char *msg) {
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

const char* wl_svm_get_last_error(void) {
  return last_error;
}

/* suppress libsvm's default print to stdout */
static void print_null(const char *s) { (void)s; }

/* ---------- dense-to-sparse conversion ---------- */

/*
 * Convert dense row-major double array to LIBSVM sparse format.
 * Pool allocation: one contiguous block of svm_node + row pointer table.
 * Skips zeros. Each row terminated with sentinel { index: -1, value: 0 }.
 *
 * Returns svm_node** (row pointers). Caller must free both
 * the row pointer array and pool (the contiguous node block).
 */
static struct svm_node** dense_to_sparse(
  const double *X, int nrow, int ncol,
  struct svm_node **pool_out
) {
  /* worst case: ncol nonzeros + sentinel per row */
  int max_nodes_per_row = ncol + 1;
  int total_nodes = nrow * max_nodes_per_row;

  struct svm_node *pool = (struct svm_node *)malloc(
    (size_t)total_nodes * sizeof(struct svm_node)
  );
  if (!pool) return NULL;

  struct svm_node **rows = (struct svm_node **)malloc(
    (size_t)nrow * sizeof(struct svm_node *)
  );
  if (!rows) { free(pool); return NULL; }

  struct svm_node *p = pool;
  for (int i = 0; i < nrow; i++) {
    rows[i] = p;
    for (int j = 0; j < ncol; j++) {
      double val = X[i * ncol + j];
      if (val != 0.0) {
        p->index = j + 1;  /* 1-based */
        p->value = val;
        p++;
      }
    }
    p->index = -1;  /* sentinel */
    p->value = 0;
    p++;
  }

  if (pool_out) *pool_out = pool;
  return rows;
}

/* convert single dense row to sparse (reusable scratch buffer) */
static void dense_row_to_sparse(
  const double *row, int ncol,
  struct svm_node *buf
) {
  struct svm_node *p = buf;
  for (int j = 0; j < ncol; j++) {
    if (row[j] != 0.0) {
      p->index = j + 1;
      p->value = row[j];
      p++;
    }
  }
  p->index = -1;
  p->value = 0;
}

/* ---------- train ---------- */

struct svm_model* wl_svm_train(
  const double *X, int nrow, int ncol,
  const double *y,
  int svm_type, int kernel_type, int degree,
  double gamma, double coef0, double C, double nu, double eps, double p,
  int shrinking, int probability, double cache_size
) {
  svm_set_print_string_function(&print_null);
  last_error[0] = '\0';

  if (!X || !y || nrow <= 0 || ncol <= 0) {
    set_error("Invalid input: X, y, nrow, ncol required");
    return NULL;
  }

  struct svm_node *pool = NULL;
  struct svm_node **sparse_X = dense_to_sparse(X, nrow, ncol, &pool);
  if (!sparse_X) {
    set_error("Failed to allocate sparse matrix");
    return NULL;
  }

  struct svm_problem prob;
  prob.l = nrow;
  prob.y = (double *)y;
  prob.x = sparse_X;

  struct svm_parameter param;
  memset(&param, 0, sizeof(param));
  param.svm_type = svm_type;
  param.kernel_type = kernel_type;
  param.degree = degree;
  param.gamma = gamma > 0 ? gamma : 1.0 / ncol;  /* default: 1/n_features */
  param.coef0 = coef0;
  param.C = C > 0 ? C : 1.0;
  param.nu = nu > 0 ? nu : 0.5;
  param.eps = eps > 0 ? eps : 0.001;
  param.p = p;
  param.shrinking = shrinking;
  param.probability = probability;
  param.cache_size = cache_size > 0 ? cache_size : 100;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;

  const char *err = svm_check_parameter(&prob, &param);
  if (err) {
    set_error(err);
    free(pool);
    free(sparse_X);
    return NULL;
  }

  struct svm_model *model = svm_train(&prob, &param);

  free(pool);
  free(sparse_X);

  if (!model) {
    set_error("Training failed");
    return NULL;
  }

  return model;
}

/* ---------- predict (batch) ---------- */

int wl_svm_predict(
  const struct svm_model *m, const double *X, int nrow, int ncol, double *out
) {
  if (!m || !X || !out) {
    set_error("predict: null argument");
    return -1;
  }

  /* scratch buffer: ncol + sentinel */
  int buf_size = ncol + 1;
  struct svm_node *buf = (struct svm_node *)malloc(
    (size_t)buf_size * sizeof(struct svm_node)
  );
  if (!buf) {
    set_error("predict: allocation failed");
    return -1;
  }

  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, buf);
    out[i] = svm_predict(m, buf);
  }

  free(buf);
  return 0;
}

int wl_svm_predict_probability(
  const struct svm_model *m, const double *X, int nrow, int ncol, double *out
) {
  if (!m || !X || !out) {
    set_error("predict_probability: null argument");
    return -1;
  }

  if (!svm_check_probability_model(m)) {
    set_error("predict_probability: model does not support probability estimates (train with probability=1)");
    return -1;
  }

  int nr_class = svm_get_nr_class(m);
  int buf_size = ncol + 1;
  struct svm_node *buf = (struct svm_node *)malloc(
    (size_t)buf_size * sizeof(struct svm_node)
  );
  if (!buf) {
    set_error("predict_probability: allocation failed");
    return -1;
  }

  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, buf);
    svm_predict_probability(m, buf, out + i * nr_class);
  }

  free(buf);
  return 0;
}

int wl_svm_predict_values(
  const struct svm_model *m, const double *X, int nrow, int ncol,
  double *out, int *out_dim
) {
  if (!m || !X || !out) {
    set_error("predict_values: null argument");
    return -1;
  }

  int nr_class = svm_get_nr_class(m);
  int svm_type = svm_get_svm_type(m);

  /* Determine output dimension per row */
  int dim;
  if (svm_type == ONE_CLASS || svm_type == EPSILON_SVR || svm_type == NU_SVR) {
    dim = 1;
  } else if (nr_class == 2) {
    dim = 1;
  } else {
    dim = nr_class * (nr_class - 1) / 2;
  }

  if (out_dim) *out_dim = dim;

  int buf_size = ncol + 1;
  struct svm_node *buf = (struct svm_node *)malloc(
    (size_t)buf_size * sizeof(struct svm_node)
  );
  if (!buf) {
    set_error("predict_values: allocation failed");
    return -1;
  }

  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, buf);
    svm_predict_values(m, buf, out + i * dim);
  }

  free(buf);
  return 0;
}

/* ---------- model I/O (MEMFS buffer) ---------- */

static int save_counter = 0;

int wl_svm_save_model(
  const struct svm_model *m, char **out_buf, int *out_len
) {
  if (!m || !out_buf || !out_len) {
    set_error("save_model: null argument");
    return -1;
  }

  char path[64];
  snprintf(path, sizeof(path), "/tmp/wl_svm_%d", save_counter++);

  int ret = svm_save_model(path, m);
  if (ret != 0) {
    set_error("save_model: write failed");
    return -1;
  }

  FILE *f = fopen(path, "rb");
  if (!f) {
    set_error("save_model: cannot read back temp file");
    remove(path);
    return -1;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *buf = (char *)malloc((size_t)size);
  if (!buf) {
    fclose(f);
    remove(path);
    set_error("save_model: allocation failed");
    return -1;
  }

  fread(buf, 1, (size_t)size, f);
  fclose(f);
  remove(path);

  *out_buf = buf;
  *out_len = (int)size;
  return 0;
}

struct svm_model* wl_svm_load_model(const char *buf, int len) {
  if (!buf || len <= 0) {
    set_error("load_model: invalid buffer");
    return NULL;
  }

  char path[64];
  snprintf(path, sizeof(path), "/tmp/wl_svm_%d", save_counter++);

  FILE *f = fopen(path, "wb");
  if (!f) {
    set_error("load_model: cannot create temp file");
    return NULL;
  }

  fwrite(buf, 1, (size_t)len, f);
  fclose(f);

  struct svm_model *m = svm_load_model(path);
  remove(path);

  if (!m) {
    set_error("load_model: failed to parse model");
    return NULL;
  }

  return m;
}

/* ---------- buffer management ---------- */

void wl_svm_free_buffer(void *ptr) {
  free(ptr);
}

void wl_svm_free_model(struct svm_model *m) {
  if (m) {
    svm_free_and_destroy_model(&m);
  }
}

/* ---------- model inspection ---------- */

int wl_svm_get_nr_class(const struct svm_model *m) {
  return m ? svm_get_nr_class(m) : 0;
}

int wl_svm_get_labels(const struct svm_model *m, int *out) {
  if (!m || !out) return -1;
  svm_get_labels(m, out);
  return 0;
}

int wl_svm_get_sv_count(const struct svm_model *m) {
  return m ? svm_get_nr_sv(m) : 0;
}

#ifdef __cplusplus
}
#endif
