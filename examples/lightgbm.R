library(future)
library(modeltuning)
library(lightgbm)
library(yardstick)

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label, free_raw_data = FALSE)
dtest <- lgb.Dataset(agaricus.test$data, label = agaricus.test$label, free_raw_data = FALSE)

# Quick function to identify the optimal number of boosting iterations with CV
# and train a LightGBM model with those optimal iterations
lgb_cv <- function(
  data,
  metric,
  label = NULL,
  nfold = 5,
  nrounds = 100,
  early_stopping_rounds = 5,
  verbose = 0,
  ...
) {
  if (!is.null(label)) {
    d <- lightgbm::lgb.Dataset(data, label = label)
  } else {
    d <- data
  }
  cv <- lightgbm::lgb.cv(
    data = d,
    params = append(list(), list(metric = metric, ...)),
    nfold = nfold,
    nrounds = nrounds,
    early_stopping_rounds = early_stopping_rounds,
    verbose = verbose
  )
  best_nrounds <- cv$best_iter
  booster <- lightgbm::lightgbm(
    data = d,
    label = label,
    params = append(list(), list(...)),
    nrounds = best_nrounds,
    verbose = verbose
  )
  return(booster)
}

# Quick function to pass in `params` as additional arguments
lgbm <- function(data, label = NULL, lightgbm_args = list(), ...) {
  params <- list(...)
  lightgbm_args$params <- append(lightgbm_args$params, params)
  lightgbm_args$data <- data
  lightgbm_args$label <- label
  booster <- do.call(lightgbm::lightgbm, lightgbm_args)
  return(booster)
}

# Grid Search (with internal cross validation) --------------------------------

## We can use the standard R matrix format

# Tune LightGBM to find the optimal values of the
# `learning_rate` and `num_leaves` parameters.
lgb_cv_grid <- GridSearch$new(
  learner = lgb_cv,
  tune_params = list(
    learning_rate = c(0.075, 0.1, 0.125),
    num_leaves = c(25, 31, 37)
  ),
  learner_args = list(
    label = agaricus.train$label, # Specify the binary response vector as a learner argument
    metric = list("auc", "binary_logloss"),
    nrounds = 30,
    nfold = 5,
    objective = "binary",
    verbose = -1
  ),
  evaluation_data = list(
    x = agaricus.test$data,
    y = factor(agaricus.test$label)
  ),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)
lgb_cv_grid_fitted <- lgb_cv_grid$fit(
  data = agaricus.train$data,
  progress = TRUE
)

# Show optimal parameters
lgb_cv_grid_fitted$best_params

# Get optimal model
lgb_cv_grid_fitted$best_model

# Get optimal metrics
lgb_cv_grid_fitted$best_metric

# Running this in parallel works great as always
#
# Side-note: if you write a wrapper function (e.g. xgb_cv), you need to EXPLICITLY
# reference external functions e.g. xgboost::xgb.cv instead of xgb.cv. Otherwise
# when it runs it in parallel it may fail to register the xgboost package.
plan(multisession, workers = 3)
lgb_cv_grid_fitted <- lgb_cv_grid$fit(
  data = agaricus.train$data,
  progress = TRUE
)
plan(sequential)

## Alternatively we can use the underlying Dataset format

lgb_cv_grid <- GridSearch$new(
  learner = lgb_cv,
  tune_params = list(
    learning_rate = c(0.075, 0.1, 0.125),
    num_leaves = c(25, 31, 37)
  ),
  learner_args = list(
    metric = list("auc", "binary_logloss"),
    nrounds = 30,
    nfold = 5,
    objective = "binary",
    verbose = -1
  ),
  evaluation_data = list(
    x = agaricus.test$data,
    y = factor(agaricus.test$label)
  ),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)

# Train sequentially ...
lgb_cv_grid_fitted <- lgb_cv_grid$fit(
  data = dtrain,
  progress = TRUE
)

# Or in parallel ...
#
# Running this in parallel can run into issues if you don't set
# `free_raw_data = FALSE` as we did when creating `dtrain` and `dtest`.
plan(multisession, workers = 3)
lgb_cv_grid_fitted <- lgb_cv_grid$fit(
  data = dtrain,
  progress = TRUE
)
plan(sequential)

# Show optimal parameters
lgb_cv_grid_fitted$best_params

# Get optimal model
lgb_cv_grid_fitted$best_model

# Cross validation (with modeltuning) -----------------------------------------

## R Matrix interface

lgb_cv_model <- CV$new(
  learner = lgbm,
  learner_args = list(
    label = agaricus.train$label, # Pass in the response vector in the learner args
    lightgbm_args = list(
      nrounds = 10,
      objective = "binary",
      verbose = -1
    ),
    learning_rate = 0.1,
    num_leaves = 31
  ),
  splitter = cv_split,
  splitter_args = list(v = 5, seed = 123),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x)
)
lgb_cv_fitted <- lgb_cv_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)

# Print the mean cross-validation ROC AUC
lgb_cv_fitted$mean_metrics

# Again, parallel works great
plan(multisession, workers = 3)
lgb_cv_fitted <- lgb_cv_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)
plan(sequential)

## Dataset interface isn't supported. This is because e.g.
## subsetting dtrain[idx, ] doesn't work

# Grid Search with cross validation (with modeltuning) ------------------------

lgb_cv_grid_model <- GridSearchCV$new(
  learner = lgbm,
  tune_params = list(
    learning_rate = c(0.075, 0.1, 0.125),
    num_leaves = c(25, 31, 37)
  ),
  learner_args = list(
    label = agaricus.train$label,
    lightgbm_args = list(objective = "binary", verbose = -1)
  ),
  splitter = cv_split,
  splitter_args = list(v = 3, seed = 123),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)
lgb_cv_grid_fitted <- lgb_cv_grid_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)

# Get the best hyperparameter values
lgb_cv_grid_fitted$best_params

# Get the ROC AUC at the best hyperparameters
lgb_cv_grid_fitted$best_metric

# Aaaaand, works great in parallel
plan(list(tweak(multisession, workers = 2), tweak(multisession, workers = 3)))
lgb_cv_grid_fitted <- lgb_cv_grid_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)
plan(sequential)
