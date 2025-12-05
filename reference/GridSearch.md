# Tune Predictive Model Hyper-parameters with Grid Search

`GridSearch` allows the user to specify a Grid Search schema for tuning
predictive model hyper-parameters with complete flexibility in the
predictive model and performance metrics.

## Public fields

- `learner`:

  Predictive modeling function.

- `scorer`:

  List of performance metric functions.

- `tune_params`:

  Data.frame of full hyper-parameter grid created from `$tune_params`

## Methods

### Public methods

- [`GridSearch$fit()`](#method-GridSearch-fit)

- [`GridSearch$new()`](#method-GridSearch-new)

- [`GridSearch$clone()`](#method-GridSearch-clone)

------------------------------------------------------------------------

### Method `fit()`

`fit` tunes user-specified model hyper-parameters via Grid Search.

#### Usage

    GridSearch$fit(
      formula = NULL,
      data = NULL,
      x = NULL,
      y = NULL,
      progress = FALSE
    )

#### Arguments

- `formula`:

  An object of class [formula](https://rdrr.io/r/stats/formula.html): a
  symbolic description of the model to be fitted.

- `data`:

  An optional data frame, or other object containing the variables in
  the model. If `data` is not provided, how `formula` is handled depends
  on `$learner`.

- `x`:

  Predictor data (independent variables), alternative interface to data
  with formula.

- `y`:

  Response vector (dependent variable), alternative interface to data
  with formula.

- `progress`:

  Logical; indicating whether to print progress across cross validation
  folds.

#### Details

`fit` follows standard R modeling convention by surfacing a formula
modeling interface as well as an alternate matrix option. The user
should use whichever interface is supported by the specified `$learner`
function.

#### Returns

An object of class
[FittedGridSearch](https://www.dmolitor.com/modelselection/reference/FittedGridSearch.md).

#### Examples

    if (require(e1071) && require(rpart) && require(yardstick)) {
      iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
      iris_new$Species <- factor(iris_new$Species == "virginica")
      iris_train <- iris_new[1:100, ]
      iris_validate <- iris_new[101:150, ]

      ### Decision Tree example

      iris_grid <- GridSearch$new(
        learner = rpart::rpart,
        learner_args = list(method = "class"),
        tune_params = list(
          minsplit = seq(10, 30, by = 5),
          maxdepth = seq(20, 30, by = 2)
        ),
        evaluation_data = list(x = iris_validate[, 1:4], y = iris_validate$Species),
        scorer = list(accuracy = yardstick::accuracy_vec),
        optimize_score = "max",
        prediction_args = list(accuracy = list(type = "class"))
      )
      iris_grid_fitted <- iris_grid$fit(
        formula = Species ~ .,
        data = iris_train
      )

      ### Example with multiple metric functions

      iris_grid <- GridSearch$new(
        learner = rpart::rpart,
        learner_args = list(method = "class"),
        tune_params = list(
          minsplit = seq(10, 30, by = 5),
          maxdepth = seq(20, 30, by = 2)
        ),
        evaluation_data = list(x = iris_validate, y = iris_validate$Species),
        scorer = list(
          accuracy = yardstick::accuracy_vec,
          auc = yardstick::roc_auc_vec
        ),
        optimize_score = "max",
        prediction_args = list(
          accuracy = list(type = "class"),
          auc = list(type = "prob")
        ),
        convert_predictions = list(
          accuracy = NULL,
          auc = function(i) i[, "FALSE"]
        )
      )
      iris_grid_fitted <- iris_grid$fit(
        formula = Species ~ .,
        data = iris_train,
      )

      # Grab the best model
      iris_grid_fitted$best_model

      # Grab the best hyper-parameters
      iris_grid_fitted$best_params

      # Grab the best model performance metrics
      iris_grid_fitted$best_metric

      ### Matrix interface example - SVM

      mtcars_train <- mtcars[1:25, ]
      mtcars_eval <- mtcars[26:nrow(mtcars), ]

      mtcars_grid <- GridSearch$new(
        learner = e1071::svm,
        tune_params = list(
          degree = 2:4,
          kernel = c("linear", "polynomial")
        ),
        evaluation_data = list(x = mtcars_eval[, -1], y = mtcars_eval$mpg),
        learner_args = list(scale = TRUE),
        scorer = list(
          rmse = yardstick::rmse_vec,
          mae = yardstick::mae_vec
        ),
        optimize_score = "min"
      )
      mtcars_grid_fitted <- mtcars_grid$fit(
        x = mtcars_train[, -1],
        y = mtcars_train$mpg
      )

    }

------------------------------------------------------------------------

### Method `new()`

Create a new GridSearch object.

#### Usage

    GridSearch$new(
      learner = NULL,
      tune_params = NULL,
      evaluation_data = NULL,
      scorer = NULL,
      optimize_score = c("min", "max"),
      learner_args = NULL,
      scorer_args = NULL,
      prediction_args = NULL,
      convert_predictions = NULL
    )

#### Arguments

- `learner`:

  Function that estimates a predictive model. It is essential that this
  function support either a formula interface with `formula` and `data`
  arguments, or an alternate matrix interface with `x` and `y`
  arguments.

- `tune_params`:

  A named list specifying the arguments of `$learner` to tune.

- `evaluation_data`:

  A two-element list containing the following elements: `x`, the
  validation data to generate predicted values with; `y`, the validation
  response values to evaluate predictive performance.

- `scorer`:

  A named list of metric functions to evaluate model performance on
  `evaluation_data`. Any provided metric function must have `truth` and
  `estimate` arguments, for true outcome values and predicted outcome
  values respectively, and must return a single numeric metric value.
  The last metric function will be the one used to identify the optimal
  model from the Grid Search.

- `optimize_score`:

  One of "max" or "min"; Whether to maximize or minimize the metric
  defined in `scorer` to find the optimal Grid Search parameters.

- `learner_args`:

  A named list of additional arguments to pass to `learner`.

- `scorer_args`:

  A named list of additional arguments to pass to `scorer`.
  `scorer_args` must either be length 1 or `length(scorer)` in the case
  where different arguments are being passed to each scoring function.

- `prediction_args`:

  A named list of additional arguments to pass to `predict`.
  `prediction_args` must either be length 1 or `length(scorer)` in the
  case where different arguments are being passed to each scoring
  function.

- `convert_predictions`:

  A list of functions to convert predicted values prior to being
  evaluated by the metric functions supplied in `scorer`. This list
  should either be length 1, in which case the same function will be
  applied to all predicted values, or `length(scorer)` in which case
  each function in `convert_predictions` will correspond with each
  function in `scorer`.

#### Returns

An object of class GridSearch.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    GridSearch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
## ------------------------------------------------
## Method `GridSearch$fit`
## ------------------------------------------------

if (require(e1071) && require(rpart) && require(yardstick)) {
  iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
  iris_new$Species <- factor(iris_new$Species == "virginica")
  iris_train <- iris_new[1:100, ]
  iris_validate <- iris_new[101:150, ]

  ### Decision Tree example

  iris_grid <- GridSearch$new(
    learner = rpart::rpart,
    learner_args = list(method = "class"),
    tune_params = list(
      minsplit = seq(10, 30, by = 5),
      maxdepth = seq(20, 30, by = 2)
    ),
    evaluation_data = list(x = iris_validate[, 1:4], y = iris_validate$Species),
    scorer = list(accuracy = yardstick::accuracy_vec),
    optimize_score = "max",
    prediction_args = list(accuracy = list(type = "class"))
  )
  iris_grid_fitted <- iris_grid$fit(
    formula = Species ~ .,
    data = iris_train
  )

  ### Example with multiple metric functions

  iris_grid <- GridSearch$new(
    learner = rpart::rpart,
    learner_args = list(method = "class"),
    tune_params = list(
      minsplit = seq(10, 30, by = 5),
      maxdepth = seq(20, 30, by = 2)
    ),
    evaluation_data = list(x = iris_validate, y = iris_validate$Species),
    scorer = list(
      accuracy = yardstick::accuracy_vec,
      auc = yardstick::roc_auc_vec
    ),
    optimize_score = "max",
    prediction_args = list(
      accuracy = list(type = "class"),
      auc = list(type = "prob")
    ),
    convert_predictions = list(
      accuracy = NULL,
      auc = function(i) i[, "FALSE"]
    )
  )
  iris_grid_fitted <- iris_grid$fit(
    formula = Species ~ .,
    data = iris_train,
  )

  # Grab the best model
  iris_grid_fitted$best_model

  # Grab the best hyper-parameters
  iris_grid_fitted$best_params

  # Grab the best model performance metrics
  iris_grid_fitted$best_metric

  ### Matrix interface example - SVM

  mtcars_train <- mtcars[1:25, ]
  mtcars_eval <- mtcars[26:nrow(mtcars), ]

  mtcars_grid <- GridSearch$new(
    learner = e1071::svm,
    tune_params = list(
      degree = 2:4,
      kernel = c("linear", "polynomial")
    ),
    evaluation_data = list(x = mtcars_eval[, -1], y = mtcars_eval$mpg),
    learner_args = list(scale = TRUE),
    scorer = list(
      rmse = yardstick::rmse_vec,
      mae = yardstick::mae_vec
    ),
    optimize_score = "min"
  )
  mtcars_grid_fitted <- mtcars_grid$fit(
    x = mtcars_train[, -1],
    y = mtcars_train$mpg
  )

}
```
