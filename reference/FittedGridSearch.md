# Fitted Models across a Tuning Grid of Hyper-parameters

`FittedGridSearch` is an object containing fitted predictive models
across a tuning grid of hyper-parameters returned by `GridSearch$fit()`
as well as relevant model information such as the best performing model,
best hyper-parameters, etc.

## Public fields

- `best_idx`:

  An integer specifying the index of `$models` that contains the
  best-performing model.

- `best_metric`:

  The performance metric of the best model on the validation data.

- `best_model`:

  The best performing predictive model.

- `best_params`:

  A named list of the hyper-parameters that result in the optimal
  predictive model.

- `tune_params`:

  Data.frame of the full hyper-parameter grid.

- `models`:

  List of predictive models at every value of `$tune_params`.

- `metrics`:

  Numeric list; Cross-validation performance metrics on each fold.

- `predictions`:

  A list containing the predicted hold-out values on every fold.

## Methods

### Public methods

- [`FittedGridSearch$new()`](#method-FittedGridSearch-new)

- [`FittedGridSearch$clone()`](#method-FittedGridSearch-clone)

------------------------------------------------------------------------

### Method `new()`

Create a new FittedGridSearch object.

#### Usage

    FittedGridSearch$new(tune_params, models, metrics, predictions, optimize_score)

#### Arguments

- `tune_params`:

  Data.frame of the full hyper-parameter grid.

- `models`:

  List of predictive models at every value of `$tune_params`.

- `metrics`:

  List of performance metrics on the validation data for every model in
  `$models`.

- `predictions`:

  A list containing the predicted values on the validation data for
  every model in `$models`.

- `optimize_score`:

  Either "max" or "min" indicating whether or not the specified
  performance metric was maximized or minimized to find the optimal
  predictive model.

#### Returns

An object of class FittedGridSearch.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    FittedGridSearch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
