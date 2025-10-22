library(future)
library(modeltuning)
library(xgboost)
library(yardstick)

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label))
dtest <- with(agaricus.test, xgb.DMatrix(data, label = label))

# Quick function to identify the optimal number of boosting iterations with CV
xgb_cv <- function(data, metrics, nfold = 5, nrounds = 100, early_stopping_rounds = 5, verbose = FALSE, ...) {
  cv <- xgboost::xgb.cv(
    data = data,
    nfold = nfold,
    nrounds = nrounds,
    verbose = verbose,
    metrics = metrics,
    early_stopping_rounds = early_stopping_rounds,
    ...
  )
  best_nrounds <- cv$best_iteration
  booster <- xgboost::xgboost(
    data = data,
    nrounds = best_nrounds,
    verbose = verbose,
    ...
  )
  return(booster)
}

# Grid Search (with internal cross validation) --------------------------------

### We can use the standard R matrix format

# Tune XGBoost to find the optimal values of the
# `max_depth` and `eta` parameters.
xgb_cv_grid <- GridSearch$new(
  learner = xgb_cv,
  tune_params = list(
    max_depth = c(2, 4, 6),
    eta = c(0.1, 0.3, 1)
  ),
  learner_args = list(
    label = agaricus.train$label, # Specify the binary response vector as a learner argument
    nrounds = 30,
    nfold = 5,
    metrics = list("rmse", "auc"),
    objective = "binary:logistic",
    verbose = FALSE
  ),
  evaluation_data = list(
    x = agaricus.test$data,
    y = factor(agaricus.test$label)
  ),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)
xgb_cv_grid_fitted <- xgb_cv_grid$fit(
  data = agaricus.train$data,
  progress = TRUE
)

# Show optimal parameters
xgb_cv_grid_fitted$best_params

# Get optimal model
xgb_cv_grid_fitted$best_model

# Running this in parallel works great as always
#
# Side-note: if you write a wrapper function (e.g. xgb_cv), you need to EXPLICITLY
# reference external functions e.g. xgboost::xgb.cv instead of xgb.cv. Otherwise
# when it runs it in parallel it may fail to register the xgboost package.
plan(multisession)
xgb_cv_grid_fitted <- xgb_cv_grid$fit(
  data = agaricus.train$data,
  progress = TRUE
)
plan(sequential)

### Alternatively we can use the underlying DMatrix format

xgb_cv_grid <- GridSearch$new(
  learner = xgb_cv,
  tune_params = list(
    max_depth = c(2, 4, 6),
    eta = c(0.1, 0.3, 1)
  ),
  learner_args = list(
    nrounds = 30,
    nfold = 5,
    metrics = list("rmse", "auc"),
    objective = "binary:logistic",
    verbose = FALSE
  ),
  evaluation_data = list(
    x = dtest,
    y = factor(getinfo(dtest, "label"))
  ),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)
xgb_cv_grid_fitted <- xgb_cv_grid$fit(
  data = dtrain,
  progress = TRUE
)

# Show optimal parameters
xgb_cv_grid_fitted$best_params

# Get optimal model
xgb_cv_grid_fitted$best_model

# Running this in parallel will FAIL!!!!
#
# This is because the underlying data object is a pointer object which cannot
# be correctly serialized. See https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
# for further reading on this.
plan(multisession)
xgb_cv_grid_fitted <- xgb_cv_grid$fit(
  data = dtrain,
  progress = TRUE
)
plan(sequential)

# Cross validation (with modeltuning) -----------------------------------------

### R Matrix interface

xgb_cv_model <- CV$new(
  learner = xgboost::xgboost,
  learner_args = list(
    label = agaricus.train$label, # Pass in the response vector in the learner args
    nrounds = 10,
    objective = "binary:logistic",
    verbose = FALSE,
    max_depth = 4,
    eta = 0.3
  ),
  splitter = cv_split,
  splitter_args = list(v = 3, seed = 123),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x)
)
xgb_cv_fitted <- xgb_cv_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)

# Print the mean cross-validation ROC AUC
xgb_cv_fitted$mean_metrics

# Again, parallel works great
plan(multisession, workers = 3)
xgb_cv_fitted <- xgb_cv_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)
plan(sequential)

### DMatrix interface isn't supported. This is because e.g. dtrain[-idx, ] doesn't work as expected
### Also, it doesn't support standard subsetting args, e.g. dtrain[idx, , drop = FALSE]

# Grid Search with cross validation (with modeltuning) ------------------------

xgb_cv_grid_model <- GridSearchCV$new(
  learner = xgboost::xgboost,
  tune_params = list(
    max_depth = c(2, 4, 6),
    eta = c(0.1, 0.3, 1)
  ),
  learner_args = list(
    label = agaricus.train$label, # Pass in the response vector in the learner args
    nrounds = 10,
    objective = "binary:logistic",
    verbose = FALSE
  ),
  splitter = cv_split,
  splitter_args = list(v = 3, seed = 123),
  scorer = list("roc_auc" = roc_auc_vec),
  convert_predictions = list("roc_auc" = function(.x) 1 - .x),
  optimize_score = "max"
)
xgb_cv_grid_fitted <- xgb_cv_grid_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)

# Get the best hyperparameter values
xgb_cv_grid_fitted$best_params

# Get the ROC AUC at the best hyperparameters
xgb_cv_grid_fitted$best_metric

# Aaaaand, works great in parallel
plan(list(tweak(multisession, workers = 3), tweak(multisession, workers = 3)))
xgb_cv_grid_fitted <- xgb_cv_grid_model$fit(
  data = agaricus.train$data,
  response = "label", # Let modeltuning know which learner arg is the response vector
  convert_response = function(.x) factor(.x), # Apply any necessary post-processing
  progress = TRUE
)
plan(sequential)
