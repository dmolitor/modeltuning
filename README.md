
<!-- README.md is generated from README.Rmd. Please edit that file -->

# modeltuning <img src='man/figures/logo-no-bg.png' align="right" height="140"/>

<!-- badges: start -->

[![pkgdown.yaml](https://github.com/dmolitor/modeltuning/actions/workflows/pkgdown.yaml/badge.svg)](https://github.com/dmolitor/modeltuning/actions/workflows/pkgdown.yaml)
[![R-CMD-check](https://github.com/dmolitor/modeltuning/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/dmolitor/modeltuning/actions/workflows/R-CMD-check.yaml)
[![CRAN
status](https://www.r-pkg.org/badges/version/modeltuning)](https://CRAN.R-project.org/package=modeltuning)
<!-- badges: end -->

The goal of modeltuning is to provide common model selection and tuning
utilities in an intuitive manner. Additionally, modeltuning aims to be:

- Fairly lightweight and not force you to learn an entirely new modeling
  paradigm
- Model/type agnostic and work easily with most R modeling packages and
  various data types including data frames, standard dense matrices, and
  `Matrix` sparse matrices
- Easily parallelizable; modeltuning is built on top of the
  [`future`](https://future.futureverse.org/) package and is compatible
  with any of the (many!) available parallelization backends.

## Installation

You can install the development version of `modeltuning` with:

``` r
# install.packages("pak")
pak::pkg_install("dmolitor/modeltuning")
```

## Usage

These are simple examples that use the built-in `iris` data-set to
illustrate the basic functionality of modeltuning.

### Cross Validation

First we’ll train a binary classification Decision Tree model to predict
whether the flowers in `iris` are of Species `virginica` and we’ll
specify a 3-fold cross validation scheme with stratification by Species
to estimate our model’s true error rate.

First, let’s split our data into a train and test set.

``` r
library(future)
library(modeltuning)
library(rpart)
library(rsample)
library(yardstick)

iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
iris_new$Species <- factor(iris_new$Species == "virginica")
iris_train <- iris_new[1:100, ]
iris_test <- iris_new[101:150, ]
```

Next, we’ll define a function to generate cross validation splits.

``` r
splitter <- function(data, ...) lapply(vfold_cv(data, ...)$splits, \(.x) .x$in_id)
```

Now, let’s specify and fit a 3-fold cross validation scheme and
calculate the *F-Measure*, *Accuracy*, and *ROC AUC* as our hold-out set
evaluation metrics.

``` r
# Specify cross validation schema
iris_cv <- CV$new(
  learner = rpart,
  learner_args = list(method = "class"),
  splitter = splitter,
  splitter_args = list(v = 3, strata = Species),
  scorer = list(
    f_meas = f_meas_vec,
    accuracy = accuracy_vec,
    auc = roc_auc_vec
  ), 
  prediction_args = list(
    f_meas = list(type = "class"),
    accuracy = list(type = "class"), 
    auc = list(type = "prob")
  ),
  convert_predictions = list(
    f_meas = NULL,
    accuracy = NULL,
    auc = function(.x) .x[, "FALSE"]
  )
)

# Fit cross validated model
iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)
```

Now, let’s check our evaluation metrics averaged across folds.

``` r
iris_cv_fitted$mean_metrics
#> $f_meas
#> [1] 0.9492091
#> 
#> $accuracy
#> [1] 0.9333173
#> 
#> $auc
#> [1] 0.9304813
```

### Grid Search

Another common model-tuning method is grid search. We’ll use it to tune
the `minsplit` and `maxdepth` parameters of our decision tree. We will
choose our optimal hyper-parameters as those that maximize the ROC AUC
on the validation set.

``` r
# Specify Grid Search schema
iris_grid <- GridSearch$new(
  learner = rpart,
  learner_args = list(method = "class"),
  tune_params = list(
    minsplit = seq(10, 30, by = 5),
    maxdepth = seq(20, 30, by = 2)
  ),
  evaluation_data = list(x = iris_test, y = iris_test$Species),
  scorer = list(
    accuracy = accuracy_vec,
    auc = roc_auc_vec
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

# Fit models across grid
iris_grid_fitted <- iris_grid$fit(
  formula = Species ~ .,
  data = iris_train
)
```

Let’s check out the optimal decision tree hyperparameters.

``` r
iris_grid_fitted$best_params
#> $minsplit
#> [1] 10
#> 
#> $maxdepth
#> [1] 20
```

### Grid Search with cross validation

Finally, `modeltuning` supports model-tuning with Grid Search using
cross validation to estimate each model’s true error rate instead of a
hold-out validation set. We’ll use cross validation to tune the same
parameters as above.

``` r
# Specify Grid Search schema with cross validation
iris_grid_cv <- GridSearchCV$new(
  learner = rpart,
  learner_args = list(method = "class"),
  tune_params = list(
    minsplit = seq(10, 30, by = 5),
    maxdepth = seq(20, 30, by = 2)
  ),
  splitter = splitter,
  splitter_args = list(v = 3, strata = Species),
  scorer = list(
    accuracy = accuracy_vec,
    auc = roc_auc_vec
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

# Fit models across grid
iris_grid_cv_fitted <- iris_grid_cv$fit(
  formula = Species ~ .,
  data = iris_train
)
```

Let’s check out the optimal decision tree hyperparameters

``` r
iris_grid_cv_fitted$best_params
#> $minsplit
#> [1] 10
#> 
#> $maxdepth
#> [1] 28
```

as well as the cross validation ROC AUC for those parameters

``` r
iris_grid_cv_fitted$best_metric
#> [1] 0.9555556
```

### Parallelization

As noted above, modeltuning is built on top of the `future` package and
can utilize any parallelization method supported by the
[`future`](https://future.futureverse.org/) package when fitting
cross-validated models or tuning models with grid search. The code below
evaluates the same cross-validated binary classification model using
local parallelization.

``` r
plan(multisession)

# Fit cross validation model
iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_train)

plan(sequential)

# Model performance metrics
iris_cv_fitted$mean_metrics
#> $f_meas
#> [1] 0.9564668
#> 
#> $accuracy
#> [1] 0.939951
#> 
#> $auc
#> [1] 0.9480072
```

And voila!

## Examples

For a bunch of worked examples with common ML frameworks, check out the
`/examples` directory!
