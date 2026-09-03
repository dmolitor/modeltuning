# Predictive Models with Cross Validation

`CV` allows the user to specify a cross validation scheme with complete
flexibility in the model, data splitting function, and performance
metrics, among other essential parameters.

## Public fields

- `learner`:

  Predictive modeling function.

- `scorer`:

  List of performance metric functions.

- `splitter`:

  Function that splits data into cross validation folds.

## Methods

### Public methods

- [`CV$fit()`](#method-CV-fit)

- [`CV$new()`](#method-CV-new)

- [`CV$clone()`](#method-CV-clone)

------------------------------------------------------------------------

### Method `fit()`

`fit` performs cross validation with user-specified parameters.

#### Usage

    CV$fit(
      formula = NULL,
      data = NULL,
      x = NULL,
      y = NULL,
      response = NULL,
      convert_response = NULL,
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

- `response`:

  String; In the absence of `formula` or `y`, this specifies which
  element of `learner_args` is the response vector.

- `convert_response`:

  Function; This should be a single function that transforms the
  response vector. E.g. a function converting a numeric binary variable
  to a factor variable.

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
[FittedCV](https://www.dmolitor.com/modelselection/reference/FittedCV.md).

#### Examples

    if (require(e1071) && require(rpart) && require(yardstick)) {
      iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
      iris_new$Species <- factor(iris_new$Species == "virginica")

      ### Decision Tree Example

      iris_cv <- CV$new(
        learner = rpart::rpart,
        learner_args = list(method = "class"),
        splitter = cv_split,
        scorer = list(accuracy = yardstick::accuracy_vec),
        prediction_args = list(accuracy = list(type = "class"))
      )
      iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)

      ### Example with multiple metric functions

      iris_cv <- CV$new(
        learner = rpart::rpart,
        learner_args = list(method = "class"),
        splitter = cv_split,
        splitter_args = list(v = 3),
        scorer = list(
          f_meas = yardstick::f_meas_vec,
          accuracy = yardstick::accuracy_vec,
          roc_auc = yardstick::roc_auc_vec,
          pr_auc = yardstick::pr_auc_vec
        ),
        prediction_args = list(
          f_meas = list(type = "class"),
          accuracy = list(type = "class"),
          roc_auc = list(type = "prob"),
          pr_auc = list(type = "prob")
        ),
        convert_predictions = list(
          f_meas = NULL,
          accuracy = NULL,
          roc_auc = function(i) i[, "FALSE"],
          pr_auc = function(i) i[, "FALSE"]
        )
      )
      iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)

      # Print the mean performance metrics across CV folds
      iris_cv_fitted$mean_metrics

      # Grab the final model fitted on the full dataset
      iris_cv_fitted$model

      ### OLS Example

      mtcars_cv <- CV$new(
        learner = lm,
        splitter = cv_split,
        splitter_args = list(v = 2),
        scorer = list("rmse" = yardstick::rmse_vec, "mae" = yardstick::mae_vec)
      )

      mtcars_cv_fitted <- mtcars_cv$fit(
        formula = mpg ~ .,
        data = mtcars
      )

      ### Matrix interface example - SVM

      mtcars_x <- model.matrix(mpg ~ . - 1, mtcars)
      mtcars_y <- mtcars$mpg

      mtcars_cv <- CV$new(
        learner = e1071::svm,
        learner_args = list(scale = TRUE, kernel = "polynomial", cross = 0),
        splitter = cv_split,
        splitter_args = list(v = 3),
        scorer = list(rmse = yardstick::rmse_vec, mae = yardstick::mae_vec)
      )
      mtcars_cv_fitted <- mtcars_cv$fit(
        x = mtcars_x,
        y = mtcars_y
      )
    }

------------------------------------------------------------------------

### Method `new()`

Create a new CV object.

#### Usage

    CV$new(
      learner = NULL,
      splitter = NULL,
      scorer = NULL,
      learner_args = NULL,
      splitter_args = NULL,
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

- `splitter`:

  A function that computes cross validation folds from an input data set
  or a pre-computed list of cross validation fold indices. If `splitter`
  is a function, it must have a `data` argument for the input data, and
  it must return a list of cross validation fold indices. If `splitter`
  is a list of integers, the number of cross validation folds is
  `length(splitter)` and each element contains the indices of the data
  observations that are included in that fold.

- `scorer`:

  A named list of metric functions to evaluate model performance on each
  cross validation fold. Any provided metric function must have `truth`
  and `estimate` arguments for true outcome values and predicted outcome
  values respectively, and must return a single numeric metric value.

- `learner_args`:

  A named list of additional arguments to pass to `learner`.

- `splitter_args`:

  A named list of additional arguments to pass to `splitter`.

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

An object of class CV.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CV$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r

## ------------------------------------------------
## Method `CV$fit`
## ------------------------------------------------

if (require(e1071) && require(rpart) && require(yardstick)) {
  iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
  iris_new$Species <- factor(iris_new$Species == "virginica")

  ### Decision Tree Example

  iris_cv <- CV$new(
    learner = rpart::rpart,
    learner_args = list(method = "class"),
    splitter = cv_split,
    scorer = list(accuracy = yardstick::accuracy_vec),
    prediction_args = list(accuracy = list(type = "class"))
  )
  iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)

  ### Example with multiple metric functions

  iris_cv <- CV$new(
    learner = rpart::rpart,
    learner_args = list(method = "class"),
    splitter = cv_split,
    splitter_args = list(v = 3),
    scorer = list(
      f_meas = yardstick::f_meas_vec,
      accuracy = yardstick::accuracy_vec,
      roc_auc = yardstick::roc_auc_vec,
      pr_auc = yardstick::pr_auc_vec
    ),
    prediction_args = list(
      f_meas = list(type = "class"),
      accuracy = list(type = "class"),
      roc_auc = list(type = "prob"),
      pr_auc = list(type = "prob")
    ),
    convert_predictions = list(
      f_meas = NULL,
      accuracy = NULL,
      roc_auc = function(i) i[, "FALSE"],
      pr_auc = function(i) i[, "FALSE"]
    )
  )
  iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)

  # Print the mean performance metrics across CV folds
  iris_cv_fitted$mean_metrics

  # Grab the final model fitted on the full dataset
  iris_cv_fitted$model

  ### OLS Example

  mtcars_cv <- CV$new(
    learner = lm,
    splitter = cv_split,
    splitter_args = list(v = 2),
    scorer = list("rmse" = yardstick::rmse_vec, "mae" = yardstick::mae_vec)
  )

  mtcars_cv_fitted <- mtcars_cv$fit(
    formula = mpg ~ .,
    data = mtcars
  )

  ### Matrix interface example - SVM

  mtcars_x <- model.matrix(mpg ~ . - 1, mtcars)
  mtcars_y <- mtcars$mpg

  mtcars_cv <- CV$new(
    learner = e1071::svm,
    learner_args = list(scale = TRUE, kernel = "polynomial", cross = 0),
    splitter = cv_split,
    splitter_args = list(v = 3),
    scorer = list(rmse = yardstick::rmse_vec, mae = yardstick::mae_vec)
  )
  mtcars_cv_fitted <- mtcars_cv$fit(
    x = mtcars_x,
    y = mtcars_y
  )
}
#> Loading required package: e1071
#> Loading required package: rpart
#> Loading required package: yardstick
```
