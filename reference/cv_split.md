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
#>  $ : int [1:7] 3 5 11 13 18 19 21
#>  $ : int [1:7] 8 16 22 25 29 30 31
#>  $ : int [1:6] 2 4 6 9 10 12
#>  $ : int [1:6] 14 20 23 24 28 32
#>  $ : int [1:6] 1 7 15 17 26 27
```
