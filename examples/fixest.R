library(fixest)
library(future)
library(modeltuning)
library(yardstick)

mtcars$w <- abs(rnorm(nrow(mtcars)))

# Cross validation ------------------------------------------------------------

mtcars_cv <- CV$new(
  learner = feglm,
  learner_args = list(
    fml = mpg ~ hp,
    family = "gaussian",
    weights = .data$w,
    cluster = ~ cyl
  ),
  splitter = cv_split,
  splitter_args = list(v = 2),
  scorer = list("rmse" = rmse_vec)
)
mtcars_cv_fitted <- mtcars_cv$fit(
  data = mtcars,
  response = function(.x) .x$mpg,
  progress = TRUE
)

# Check the CV error
mtcars_cv_fitted$mean_metrics

# Fit in parallel
plan(multisession, workers = 2)
mtcars_cv_fitted <- mtcars_cv$fit(
  data = mtcars,
  response = function(.x) .x$mpg,
  progress = TRUE
)
plan(sequential)

# Grid Search -----------------------------------------------------------------

mtcars_train <- mtcars[1:25, ]
mtcars_eval <- mtcars[26:nrow(mtcars), ]

mtcars_grid <- GridSearch$new(
  learner = feglm,
  tune_params = list(
    lean = c(TRUE, FALSE),
    family = c("gaussian", "poisson")
  ),
  learner_args = list(
    fml = mpg ~ hp,
    weights = mtcars_train$w,
    cluster = ~ cyl
  ),
  evaluation_data = list(x = mtcars_eval, y = mtcars_eval$mpg),
  scorer = list("rmse" = rmse_vec)
)
mtcars_grid_fitted <- mtcars_grid$fit(data = mtcars_train, progress = TRUE)

# Get best params
mtcars_grid_fitted$best_params

# Get best metrics
mtcars_grid_fitted$best_metric

# In parallel
plan(multisession, workers = 4)
mtcars_grid_fitted <- mtcars_grid$fit(data = mtcars_train, progress = TRUE)
plan(sequential)

# Grid Search with cross validation -------------------------------------------

mtcars_grid_cv <- GridSearchCV$new(
  learner = feglm,
  tune_params = list(
    lean = c(TRUE, FALSE),
    family = c("gaussian", "poisson")
  ),
  learner_args = list(
    fml = mpg ~ hp,
    weights = .data$w,
    cluster = ~ cyl
  ),
  splitter = cv_split,
  splitter_args = list(v = 2),
  scorer = list("rmse" = rmse_vec)
)
mtcars_grid_cv_fitted <- mtcars_grid_cv$fit(
  data = mtcars,
  response = function(.x) .x$mpg,
  progress = TRUE
)

# Get best params
mtcars_grid_cv_fitted$best_params

# In parallel!
plan(list(tweak(multisession, workers = 4), tweak(multisession, workers = 2)))
mtcars_grid_cv_fitted <- mtcars_grid_cv$fit(
  data = mtcars,
  response = function(.x) .x$mpg,
  progress = TRUE
)
plan(sequential)
