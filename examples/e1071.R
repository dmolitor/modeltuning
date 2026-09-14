### SVM classification example with the e1071 package ###

library(e1071)
library(future)
library(modeltuning)
library(yardstick)

iris_new <- iris
iris_new$Species <- factor(as.integer(iris_new$Species == "virginica"))
iris_new <- iris_new[sample(1:nrow(iris_new)), ]

# Grid Search -----------------------------------------------------------------

iris_train <- iris_new[1:100, ]
iris_eval <- iris_new[101:nrow(iris_new), ]

iris_grid <- GridSearch$new(
  learner = svm,
  tune_params = list(
    kernel = c("linear", "polynomial"),
    degree = c(3, 5),
    cost = c(0.5, 1)
  ),
  learner_args = list(type = "C-classification", probability = TRUE),
  evaluation_data = list(x = iris_eval[, -5], y = iris_eval$Species),
  scorer = list(
    "accuracy" = accuracy_vec,
    "roc_auc" = roc_auc_vec
  ),
  prediction_args = list(
    "accuracy" = NULL,
    "roc_auc" = list(probability = TRUE)
  ),
  convert_predictions = list(
    "accuracy" = NULL,
    "roc_auc" = function(.x) attr(.x, "probabilities")[, "0"]
  ),
  optimize_score = "max"
)
iris_grid_fitted <- iris_grid$fit(
  formula = Species ~ .,
  data = iris_train,
  progress = TRUE
)

# Grab best model
iris_grid_fitted$best_model

# Grab best metric
iris_grid_fitted$best_metric

# In parallel
plan(multisession, workers = 3)
iris_grid_fitted <- iris_grid$fit(
  formula = Species ~ .,
  data = iris_train,
  progress = TRUE
)
plan(sequential)

# Cross validation ------------------------------------------------------------

iris_cv <- CV$new(
  learner = svm,
  learner_args = list(
    kernel = "linear",
    cost = 0.5,
    type = "C-classification",
    probability = TRUE
  ),
  splitter = cv_split,
  splitter_args = list(v = 3),
  scorer = list(
    "accuracy" = accuracy_vec,
    "roc_auc" = roc_auc_vec
  ),
  prediction_args = list(
    "accuracy" = NULL,
    "roc_auc" = list(probability = TRUE)
  ),
  convert_predictions = list(
    "accuracy" = NULL,
    "roc_auc" = function(.x) attr(.x, "probabilities")[, "0"]
  )
)
iris_cv_fitted <- iris_cv$fit(
  formula = Species ~ .,
  data = iris_new,
  progress = TRUE
)

# Mean performance metrics
iris_cv_fitted$mean_metrics

# Get the fully fitted model
iris_cv_fitted$model

# In parallel
plan(multisession, workers = 3)
iris_cv_fitted <- iris_cv$fit(
  formula = Species ~ .,
  data = iris_new,
  progress = TRUE
)
plan(sequential)

# Grid search with cross validation -------------------------------------------

iris_grid_cv <- GridSearchCV$new(
  learner = svm,
  tune_params = list(
    kernel = c("linear", "polynomial"),
    degree = c(3, 5),
    cost = c(0.5, 1)
  ),
  learner_args = list(type = "C-classification", probability = TRUE),
  splitter = cv_split,
  splitter_args = list(v = 3),
  scorer = list(
    "accuracy" = accuracy_vec,
    "roc_auc" = roc_auc_vec
  ),
  prediction_args = list(
    "accuracy" = NULL,
    "roc_auc" = list(probability = TRUE)
  ),
  convert_predictions = list(
    "accuracy" = NULL,
    "roc_auc" = function(.x) attr(.x, "probabilities")[, "0"]
  ),
  optimize_score = "max"
)
iris_grid_cv_fitted <- iris_grid_cv$fit(
  formula = Species ~ .,
  data = iris_new,
  progress = TRUE
)

# Grab best model
iris_grid_cv_fitted$best_model

# Grab best metric
iris_grid_cv_fitted$best_metric

# Get best parameter values
iris_grid_cv_fitted$best_params

# Aaaaand, works great in parallel
plan(list(tweak(multisession, workers = 2), tweak(multisession, workers = 3)))
iris_grid_cv_fitted <- iris_grid_cv$fit(
  formula = Species ~ .,
  data = iris_new,
  progress = TRUE
)
plan(sequential)