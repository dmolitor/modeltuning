# Fitted, Cross-Validated Predictive Models

`FittedCV` is a fitted, cross-validated predictive model object that is
returned by `CV$fit()` and contains relevant model components,
cross-validation metrics, validation set predicted values, etc.

## Public fields

- `folds`:

  A list of length `$nfolds` where each element contains the indices of
  the observations contained in that fold.

- `model`:

  Predictive model fitted on the full data set.

- `mean_metrics`:

  Numeric list; Cross-validation performance metrics averaged across
  folds.

- `metrics`:

  Numeric list; Cross-validation performance metrics on each fold.

- `nfolds`:

  An integer specifying the number of cross-validation folds.

- `predictions`:

  A list containing the predicted hold-out values on every fold.

## Methods

### Public methods

- [`FittedCV$new()`](#method-FittedCV-new)

- [`FittedCV$clone()`](#method-FittedCV-clone)

------------------------------------------------------------------------

### Method `new()`

Create a new FittedCV object.

#### Usage

    FittedCV$new(folds, model, metrics, nfolds, predictions)

#### Arguments

- `folds`:

  A list of length `$nfolds` where each element contains the indices of
  the observations contained in that fold.

- `model`:

  Predictive model fitted on the full data set.

- `metrics`:

  Numeric list; Cross-validation performance metrics on each fold.

- `nfolds`:

  An integer specifying the number of cross-validation folds.

- `predictions`:

  A list containing the predicted hold-out values on every fold.

#### Returns

An object of class FittedCV.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    FittedCV$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
