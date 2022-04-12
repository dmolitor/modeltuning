
<!-- README.md is generated from README.Rmd. Please edit that file -->

# modelselection

<!-- badges: start -->
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

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(modelselection)
## basic example code
```
