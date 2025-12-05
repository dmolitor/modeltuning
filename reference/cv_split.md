# Generate cross-validation fold indices

Splits row indices of a data frame or matrix into `k` folds for
cross-validation.

## Usage

``` r
cv_split(data, v = 5, seed = NULL)
```

## Arguments

- data:

  A data frame or matrix.

- v:

  Integer. Number of folds. Defaults to 5.

- seed:

  Optional integer. Random seed for reproducibility.

## Value

A list of length `v`, where each element is a vector of row indices for
that fold.

## Examples

``` r
folds <- cv_split(mtcars, v = 5)
str(folds)
#> List of 5
#>  $ : int [1:7] 4 5 10 12 15 18 29
#>  $ : int [1:7] 11 13 17 19 20 26 31
#>  $ : int [1:6] 2 3 16 22 24 30
#>  $ : int [1:6] 6 9 21 27 28 32
#>  $ : int [1:6] 1 7 8 14 23 25
```
