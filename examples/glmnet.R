library(doFuture)
library(future)
library(glmnet)
library(modeltuning)
library(yardstick)

data(QuickStartExample, package = "glmnet")

# Grid Search (with internal cross validation) --------------------------------

x_train <- QuickStartExample$x[1:75, ]
y_train <- QuickStartExample$y[1:75]

x_eval <- QuickStartExample$x[76:100, ]
y_eval <- QuickStartExample$y[76:100]

# Tune glmnet to find the optimal penalization parameter `alpha`
glmnet_grid <- GridSearch$new(
  learner = cv.glmnet,
  tune_params = list(alpha = seq(0, 1, by = 0.1), relax = c(TRUE, FALSE)),
  learner_args = list(nfolds = 5, parallel = TRUE),
  evaluation_data = list(x = x_eval, y = y_eval),
  scorer = list("rmse" = rmse_vec),
  convert_predictions = list("rmse" = drop),
  optimize_score = "min"
)
glmnet_grid_fitted <- glmnet_grid$fit(
  x = x_train,
  y = y_train,
  progress = TRUE
)

# Get optimal parameters
glmnet_grid_fitted$best_params

# Get best model
glmnet_grid_fitted$best_model

# Get best metrics
glmnet_grid_fitted$best_metric

# Running this in parallel works great as always. We need to specify
# Two parallel layers to parallelize BOTH the grid search AND cross-validation
plan(list(tweak(multicore, workers = 2), tweak(multicore, workers = 5)))
glmnet_grid_fitted <- glmnet_grid$fit(
  x = x_train,
  y = y_train,
  progress = TRUE
)
plan(sequential)

# Cross validation ------------------------------------------------------------

glmnet_cv <- CV$new(
  learner = glmnet,
  learner_args = list(family = "gaussian", alpha = 1),
  splitter = cv_split,
  scorer = list("rmse" = rmse_vec),
  prediction_args = list("rmse" = list(s = 0.1)),
  convert_predictions = list("rmse" = drop)
)
glmnet_cv_fitted <- glmnet_cv$fit(
  x = QuickStartExample$x,
  y = QuickStartExample$y,
  progress = TRUE
)

# Grid search with cross validation --------------------------------------------

## Warning: This is actually doing something glmnet does NOT recommend!
## This is purely for demonstration.
glmnet_grid_cv <- GridSearchCV$new(
  learner = glmnet,
  tune_params = list(lambda = log(seq(1, 3, by = .02))),
  learner_args = list(family = "gaussian", alpha = 1),
  splitter = cv_split,
  scorer = list("rmse" = rmse_vec),
  prediction_args = list("rmse" = list(s = 0.1)),
  convert_predictions = list("rmse" = drop)
)
glmnet_grid_cv_fitted <- glmnet_grid_cv$fit(
  x = QuickStartExample$x,
  y = QuickStartExample$y,
  progress = TRUE
)

# Get optimal parameters
glmnet_grid_cv_fitted$best_params

# Get best metric
glmnet_grid_cv_fitted$best_metric

# Run in parallel
plan(multisession)
glmnet_grid_cv_fitted <- glmnet_grid_cv$fit(
  x = QuickStartExample$x,
  y = QuickStartExample$y,
  progress = TRUE
)
plan(sequential)
