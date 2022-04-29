get_namespace_name <- function(fn) {
  tryCatch(
    unname(getNamespaceName(environment(fn))),
    error = function(e) NULL
  )
}

modelselection_fns <- function() {
  c(
    "CV",
    "FittedCV",
    "FittedGridSearch",
    "FittedGridSearchCV",
    "get_namespace_name",
    "GridSearch",
    "GridSearchCV",
    "modelselection_fns"
  )
}
