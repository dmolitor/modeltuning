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

validate_scorer <- function(scorer, scorer_args, prediction_args) {
  if (is.null(scorer)){
    abort(c("Missing argument:", "x" = "`scorer` must be specified"))
  }
  if (is.null(names(scorer)) || any(vapply(names(scorer), function(i) i == "", NA))) {
    abort(c("Missing attribute names:", "x" = "Each element of `scorer` must have a name"))
  }
  if (is.function(scorer)) {
    if (!all(c("truth", "estimate") %in% names(formals(scorer)))) {
      abort(c("Improper `scorer` function:", "x" = "`scorer` functions must have `truth` and `estimate` arguments for true outcomes and predicted outcomes."))
    }
  } else if (is.list(scorer)) {
    formals_good <- vapply(scorer, function(x) all(c("truth", "estimate") %in% names(formals(x))), logical(1))
    if (!all(formals_good)) {
      abort(c("Improper `scorer` function:", "x" = "All `scorer` functions must have `truth` and `estimate` arguments for true outcomes and predicted outcomes."))
    }
  }
  if (!identical(names(scorer), names(scorer_args)) && !is.null(scorer_args) && length(scorer_args) != 1) {
    abort(
      c(
        "Missing attribute names:",
        "x" = paste0(
          "The following elements in `scorer` are missing in `scorer_args`: ",
          if (length(setdiff(names(scorer), names(scorer_args))) > 5) {
            paste0(
              c(setdiff(names(scorer), names(scorer_args))[1:5], "..."),
              collapse = ", "
            )
          } else {
            paste0(
              setdiff(names(scorer), names(scorer_args)),
              collapse = ", "
            )
          }
        ),
        "x" = paste0(
          "The following elements in `scorer_args` are missing in `scorer`: ",
          if (length(setdiff(names(scorer_args), names(scorer))) > 5) {
            paste0(
              c(setdiff(names(scorer_args), names(scorer))[1:5], "..."),
              collapse = ", "
            )
          } else {
            paste0(
              setdiff(names(scorer_args), names(scorer)),
              collapse = ", "
            )
          }
        )
      )
    )
  }
  if (!identical(names(scorer), names(prediction_args)) && !is.null(prediction_args) && length(prediction_args) != 1) {
    abort(
      c(
        "Missing attribute names:",
        "x" = paste0(
          "The following elements in `scorer` are missing in `prediction_args`: ",
          if (length(setdiff(names(scorer), names(prediction_args))) > 5) {
            paste0(
              c(setdiff(names(scorer), names(prediction_args))[1:5], "..."),
              collapse = ", "
            )
          } else {
            paste0(
              setdiff(names(scorer), names(prediction_args)),
              collapse = ", "
            )
          }
        ),
        "x" = paste0(
          "The following elements in `scorer_args` are missing in `scorer`: ",
          if (length(setdiff(names(prediction_args), names(scorer))) > 5) {
            paste0(
              c(setdiff(names(prediction_args), names(scorer))[1:5], "..."),
              collapse = ", "
            )
          } else {
            paste0(
              setdiff(names(prediction_args), names(scorer)),
              collapse = ", "
            )
          }
        )
      )
    )
  }
}

validate_splitter <- function(splitter) {
  if (!rlang::is_list(splitter)) {
    if (!is.function(splitter)) abort("`splitter` must be a function or a list of indices")
    if (!"data" %in% names(formals(splitter))) {
      abort(c("Improper `splitter` function:", "x" = "`splitter` must have an argument named `data`"))
    }
  } else {
    valid_splits <- vapply(splitter, rlang::is_list, logical(1))
    if (!all(valid_splits)) {
      abort(c("Invalid `splitter`:", "x" = "`splitter` should be a list of indices specifying data splits"))
    }
  }
}