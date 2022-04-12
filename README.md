
<!-- README.md is generated from README.Rmd. Please edit that file -->

# modelselection

<!-- badges: start -->

[![R-CMD-check](https://github.com/dmolitor/modelselection/workflows/R-CMD-check/badge.svg)](https://github.com/dmolitor/modelselection/actions)
<!-- badges: end -->

The goal of `modelselection` is to provide common model selection and
tuning utilities in an intuitive manner. Specifically, I want something
that is lightweight (so not `mlr3`) and doesn’t force you to adopt a
whole new modeling paradigm (so not `tidymodels`). Also, I want the
provided functionality to be very type agnostic and be able to work with
data frames, standard dense matrices, and `Matrix` sparse matrices.
Finally, I want it to be easily distributable (It’s built on top of the
`future.apply` package) and I want to have full control over it 😉.

## Installation

You can install the development version of `modelselection` with:

``` r
# install.packages("devtools")
devtools::install_github("dmolitor/modelselection")
```

## Basic Example

This is a simple example that uses the built-in `iris` data-set to
illustrate the basic functionality of `modelselection`.

``` r
library(modelselection)
library(rpart)
library(rsample)
library(yardstick)
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.

iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
iris_new$Species <- factor(iris_new$Species == "virginica")
iris_train <- iris_new[1:100, ]
iris_test <- iris_new[101:150, ]
```

We’ll use Decision Trees to train a binary classification model to
predict whether the flowers in `iris` are of Species `virginica` and
we’ll specify a 3-fold Cross-Validation scheme with stratification by
Species to estimate our model’s true error error rate.

``` r
# Specify Cross Validation schema
iris_cv <- CV$new(
  learner = rpart,
  learner_args = list(method = "class"),
  splitter = vfold_cv, 
  splitter_args = list(v = 3, strata = "Species"),
  scorer = list(
    "f_meas" = f_meas_vec,
    "accuracy" = accuracy_vec,
    "auc" = roc_auc_vec
  ), 
  prediction_args = list(
    "f_meas" = list(type = "class"),
    "accuracy" = list(type = "class"), 
    "auc" = list(type = "prob")
  ),
  convert_predictions = list(
    NULL,
    NULL,
    function(.x) .x[, "FALSE"]
  )
)
# Fit Cross Validated model
iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_train)

cat(
  "F-Measure:", paste0(round(100 * iris_cv_fitted$mean_metrics$f_meas, 2), "%"),
  "\n Accuracy:", paste0(round(100 * iris_cv_fitted$mean_metrics$accuracy, 2), "%"),
  "\n      AUC:", paste0(round(iris_cv_fitted$mean_metrics$auc, 4))
)
#> F-Measure: 95.27% 
#>  Accuracy: 94% 
#>       AUC: 0.9349
```
