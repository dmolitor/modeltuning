attached_pkgs <- function() {
  sub("^package:", "", grep("^package:", search(), value = TRUE))
}

check_list_or_null <- function(...) {
  dots <- rlang::list2(...)
  lapply(
    names(dots),
    function(.x) {
      if (!is_expr_list_or_null(dots[[.x]]))
        abort(paste0("Invalid argument: ", .x, " must be either NULL or a list"))
    }
  )
}

compare_names <- function(...) {
  dots <- list(...)
  stopifnot(length(dots) == 2)
  x_name <- names(dots)[[1]]
  y_name <- names(dots)[[2]]
  if (length(dots[[2]]) == 1 && is.null(dots[[2]][[1]])) return(invisible(NULL))
  if (!identical(names(dots[[1]]), names(dots[[2]])) && !is.null(dots[[2]])) {
    abort(
      c(
        "Missing attribute names:",
        "x" = if (length(setdiff(names(dots[[1]]), names(dots[[2]]))) > 0) {
          paste0(
            paste0("The following elements in `", x_name, "` are missing in `", y_name, "`: "),
            if (length(setdiff(names(dots[[1]]), names(dots[[2]]))) > 5) {
              paste0(
                c(setdiff(names(dots[[1]]), names(dots[[2]]))[1:5], "..."),
                collapse = ", "
              )
            } else {
              paste0(
                setdiff(names(dots[[1]]), names(dots[[2]])),
                collapse = ", "
              )
            }
          )
        },
        "x" = if (length(setdiff(names(dots[[2]]), names(dots[[1]]))) > 0) {
          paste0(
            paste0("The following elements in `", y_name, "` are missing in `", x_name, "`: "),
            if (length(setdiff(names(dots[[2]]), names(dots[[1]]))) > 5) {
              paste0(
                c(setdiff(names(dots[[2]]), names(dots[[1]]))[1:5], "..."),
                collapse = ", "
              )
            } else {
              paste0(
                setdiff(names(dots[[2]]), names(dots[[1]])),
                collapse = ", "
              )
            }
          )
        }
      )
    )
  }
}

#' Generate cross-validation fold indices
#'
#' Splits row indices of a data frame or matrix into \code{k} folds for cross-validation.
#'
#' @param data A data frame or matrix.
#' @param v Integer. Number of folds. Defaults to 5.
#' @param seed Optional integer. Random seed for reproducibility.
#'
#' @return A list of length \code{v}, where each element is a vector of row indices for that fold.
#'
#' @examples
#' folds <- cv_split(mtcars, v = 5)
#' str(folds)
#'
#' @export
cv_split <- function(data, v = 5, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n <- nrow(data)
  fold_ids <- sample(rep(1:v, length.out = n))
  unname(split(seq_len(n), fold_ids))
}

extract_symbols <- function(expr) {
  expr <- rlang::get_expr(expr)
  recurse <- function(x) {
    if (rlang::is_symbol(x)) {
      as.character(x)
    } else if (is.call(x)) {
      unlist(lapply(as.list(x), recurse), use.names = FALSE)
    } else {
      NULL
    }
  }
  unique(recurse(expr))
}

expr_to_quoted_list <- function(x) {
  stopifnot(is_expr_list_or_null(x))
  if (is.null(x)) return(NULL)
  x <- lapply(x, function(.x) .x)
  x[x != "list"]
}

extract_params <- function(params, index) {
  params <- params[index, , drop = FALSE]
  set_names(lapply(params, function(.x) .x[[1]]), names(params))
}

get_namespace_name <- function(fn) {
  tryCatch(
    unname(getNamespaceName(environment(fn))),
    error = function(e) NULL
  )
}

get_objects_from_env <- function(objects, env = rlang::caller_env()) {
  found <- character()
  while (!identical(env, rlang::empty_env())) {
    present <- intersect(objects, ls(env, all.names = TRUE))
    found <- union(found, present)
    env <- rlang::env_parent(env)
  }
  if (length(found) == 0) return(NULL)
  unique(unlist(found))
}

is_expr_list_or_null <- function(x) {
  if (is.null(x)) return(TRUE)
  if (rlang::is_call(x) && rlang::call_name(x) == "list" || rlang::is_list(x)) {
    return(TRUE)
  }
  return(FALSE)
}

modelselection_fns <- function() {
  c(
    "CV",
    "extract_params",
    "FittedCV",
    "FittedGridSearch",
    "FittedGridSearchCV",
    "get_namespace_name",
    "GridSearch",
    "GridSearchCV",
    "modelselection_fns"
  )
}

set_names <- function(x, names) {
  names(x) <- names
  return(x)
}

validate_scorer <- function(scorer) {
  if (is.null(scorer)){
    abort(c("Missing argument:", "x" = "`scorer` must be specified"))
  }
  if (!rlang::is_list(scorer) || is.null(names(scorer)) || any(vapply(names(scorer), function(i) i == "", NA))) {
    abort(c("Missing attribute names:", "x" = "Each element of `scorer` must have a name"))
  }
  for (s in scorer) {
    if (!rlang::is_function(s)) abort("`scorer` must be a list of functions")
    if (!all(c("truth", "estimate") %in% names(formals(s)))) {
      abort(c("Improper `scorer` function:", "x" = "`scorer` functions must have `truth` and `estimate` arguments for true outcomes and predicted outcomes."))
    }
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