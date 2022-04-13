
<!-- README.md is generated from README.Rmd. Please edit that file -->

# modelselection

<!-- badges: start -->

[![R-CMD-check](https://github.com/dmolitor/modelselection/workflows/R-CMD-check/badge.svg)](https://github.com/dmolitor/modelselection/actions)
[![pkgdown](https://github.com/dmolitor/modelselection/workflows/pkgdown/badge.svg)](https://github.com/dmolitor/modelselection/actions)
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

## Usage

This is a simple example that uses the built-in `iris` data-set to
illustrate the basic functionality of `modelselection`.

We’ll use Decision Trees to train a binary classification model to
predict whether the flowers in `iris` are of Species `virginica` and
we’ll specify a 3-fold Cross-Validation scheme with stratification by
Species to estimate our model’s true error rate.

### Modeling Data

``` r
library(future)
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

### Specify and Fit Model

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
```

### Compare CV and Test Set Error

``` r
# Extract full model and generate test set predictions
full_model <- iris_cv_fitted$model
test_class <- predict(full_model, iris_test, type = "class")
test_probs <- predict(full_model, iris_test)[, "FALSE"]

# Calculate metrics on test set
f_meas_test <- f_meas_vec(iris_test$Species, test_class)
accuracy_test <- accuracy_vec(iris_test$Species, test_class)
auc_test <- roc_auc_vec(iris_test$Species, test_probs)

cat(
  "  CV F-Measure:", paste0(round(100 * iris_cv_fitted$mean_metrics$f_meas, 2), "%"),
  "  --  Test F-Measure:", paste0(round(100 * f_meas_test, 2), "%"),
  "\n   CV Accuracy:", paste0(round(100 * iris_cv_fitted$mean_metrics$accuracy, 2), "%"),
  " --  Test Accuracy:", paste0(round(100 * accuracy_test, 2), "%"),
  "\n        CV AUC:", paste0(round(iris_cv_fitted$mean_metrics$auc, 4)),
  " --  Test AUC:", paste0(round(auc_test, 4))
)
#>   CV F-Measure: 93.76%   --  Test F-Measure: 97.14% 
#>    CV Accuracy: 91.95%  --  Test Accuracy: 96% 
#>         CV AUC: 0.9167  --  Test AUC: 0.9375
```

### Parallelization

As noted above, `modelselection` is built on top of the `future.apply`
package and can utilize any parallelization method supported by the
[`future`](https://future.futureverse.org/) package. The code below
evaluates the same cross-validated binary classification model using
local multi-core parallelization.

``` r
# Initialize multi-core parallel strategy
plan(multisession)

# Fit Cross Validated model
iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_train)
```

And voila!
