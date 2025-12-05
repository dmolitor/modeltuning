# Fitted Models with Cross Validation across a Tuning Grid of Hyper-parameters

`FittedGridSearchCV` is an object containing fitted predictive models
across a tuning grid of hyper-parameters returned by
`GridSearchCV$fit()` as well as relevant model information such as the
best performing model, best hyper-parameters, etc.

## Public fields

- `best_idx`:

  An integer specifying the index of `$models` that contains the
  best-performing model.

- `best_metric`:

  The average performance metric of the best model across
  cross-validation folds.

- `best_model`:

  The best performing predictive model.

- `best_params`:

  A named list of the hyper-parameters that result in the optimal
  predictive model.

- `folds`:

  A list of length `$models` where each element contains a list of the
  cross-validation indices for each fold.

- `tune_params`:

  A [data.frame](https://rdrr.io/r/base/data.frame.html) of the full
  hyper-parameter grid.

- `models`:

  List of predictive models at every value of `$tune_params`.

- `metrics`:

  Numeric list; Cross-validation performance metrics for every model in
  `$models`.

- `predictions`:

  A list containing the cross-validation fold predictions for each model
  in `$models`.

## Methods

### Public methods

- [`FittedGridSearchCV$new()`](#method-FittedGridSearchCV-new)

- [`FittedGridSearchCV$clone()`](#method-FittedGridSearchCV-clone)

------------------------------------------------------------------------

### Method `new()`

Create a new FittedGridSearchCV object.

#### Usage

    FittedGridSearchCV$new(
      tune_params,
      models,
      folds,
      metrics,
      predictions,
      optimize_score
    )

#### Arguments

- `tune_params`:

  Data.frame of the full hyper-parameter grid.

- `models`:

  List of predictive models at every value of `$tune_params`.

- `folds`:

  List of cross-validation indices at every value of `$tune_params`.

- `metrics`:

  List of cross-validation performance metrics for every model in
  `$models`.

- `predictions`:

  A list containing the predicted values on the cross-validation folds
  for every model in `$models`.

- `optimize_score`:

  Either "max" or "min" indicating whether or not the specified
  performance metric was maximized or minimized to find the optimal
  predictive model.

#### Returns

An object of class FittedGridSearchCV.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    FittedGridSearchCV$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
