#' Fitted, Cross-Validated Predictive Models
#'
#' @description
#' `FittedCV` is a fitted, cross-validated predictive model object that is
#' returned by `CV$fit()` and contains relevant model components,
#' cross-validation metrics, validation set predicted values, etc.
#'
#' @export
FittedGridSearch <- R6Class(
  classname = "FittedGridSearch",
  public = list(
    #' @field folds A list of length `$nfolds` where each element contains the
    #'   indices of the observations contained in that fold.
    tune_params = NULL,
    #' @description
    #' Create a new [FittedCV] object.
    #'
    #' @param folds A list of length `$nfolds` where each element contains the
    #'   indices of the observations contained in that fold.
    #' @param model Predictive model fitted on the full data set.
    #' @param metrics Numeric list; Cross-validation performance metrics on each
    #'   fold.
    #' @param nfolds An integer specifying the number of cross-validation folds.
    #' @param predictions A list containing the predicted hold-out values on
    #'   every fold.
    #'
    #' @return An object of class [FittedCV].
    initialize = function(tune_params, models, metrics, predictions) {
      self$tune_params <- tune_params
      self$models <- models
      self$metrics <- metrics
      self$predictions <- predictions
    },
    #' @field model Predictive model fitted on the full data set.
    models = NULL,
    #' @field metrics Numeric list; Cross-validation performance metrics on each
    #'   fold.
    metrics = NULL,
    #' @field predictions A list containing the predicted hold-out values on
    #'   every fold.
    predictions = NULL
  )
)

#' Tune Predictive Model Hyperparameters with Grid Search
#'
#' @description
#' `GridSearch` allows the user to specify a Grid Search schema for tuning
#' predictive model hyperparameters with complete flexibility in the predictive
#' model and performance metrics
GridSearch <- R6Class(
  classname = "GridSearch",
  public = list(
    fit = function(formula = NULL,
                   data = NULL,
                   x = NULL,
                   y = NULL,
                   progress = FALSE) {
      input <- private$check_data_input(formula, data, x, y)
      response_var <- private$response(private$evaluation_data$y)
      if (progress) {
        with_progress({
          model_output <- private$fit_grid(input, response_var)
        })
      } else {
        model_output <- private$fit_grid(input, response_var)
      }
      FittedGridSearch$new(
        tune_params = self$tune_params,
        models = model_output$models,
        metrics = model_output$metrics,
        predictions = model_output$preds
      )
    },
    initialize = function(learner = NULL,
                          tune_params = NULL,
                          evaluation_data = NULL,
                          scorer = NULL,
                          optimize_score = c("min", "max"),
                          learner_args = NULL,
                          scorer_args = NULL,
                          prediction_args = NULL,
                          convert_predictions = NULL) {
      if (is.null(enexpr(learner))){
        abort(
          c(
            "Missing argument:",
            "x" = "`learner` must be specified"
          )
        )
      }
      if (is.null(tune_params)) {
        abort(
          c(
            "Missing argument:",
            "x" = "`tune_params` must be specified"
          )
        )
      }
      if (is.null(names(tune_params)) || any(vapply(names(tune_params), function(i) i == "", NA))) {
        abort(
          c(
            "Missing attribute names:",
            "x" = "Each element of `tune_params` must have a name corresponding to an argument of `learner`"
          )
        )
      }
      if (is.null(evaluation_data) || !identical(names(evaluation_data), c("x", "y"))) {
        abort(
          c(
            "Mis-specified argument:",
            "i" = "`evaluation_data` must be a named list with elements `x` and `y`"
          )
        )
      }
      if (is.null(scorer)){
        abort(
          c(
            "Missing argument:",
            "x" = "`scorer` must be specified"
          )
        )
      }
      if (is.null(names(scorer)) || any(vapply(names(scorer), function(i) i == "", NA))) {
        abort(
          c(
            "Missing attribute names:",
            "x" = "Each element of `scorer` must have a name"
          )
        )
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
      self$learner <- enexpr(learner)
      private$learner_args <- learner_args
      self$scorer <- Map(
        f = function(.x, .y) call2(.x, !!!.y),
        scorer,
        if (is.null(scorer_args)) list(NULL) else scorer_args
      )
      private$prediction_args <- if (is.null(prediction_args)) {
        list(NULL)
      } else {
        prediction_args
      }
      private$convert_predictions <- if (
        !is.null(convert_predictions) &&
        !(is.list(convert_predictions) || is.atomic(convert_predictions))
      ) {
        list(convert_predictions)
      } else {
        convert_predictions
      }
      self$tune_params <- expand.grid(tune_params)
      private$optimize_score <- match.arg(optimize_score)
      private$evaluation_data <- evaluation_data
    },
    learner = NULL,
    scorer = NULL,
    tune_params = NULL
  ),
  private = list(
    check_data_input = function(formula = NULL, data = NULL, x = NULL, y = NULL) {
      if (all(vapply(list(formula, data, x, y), is.null, NA))) {
        abort(
          c(
            "Missing data elements:",
            "x" = "No data elements were provided",
            "i" = "Either `data` and/or `formula` or `x` and `y` must be supplied"
          )
        )
      }
      if (any(vapply(list(x, y), is.null, NA))) {
        if (is.null(formula)) {
          abort(
            c(
              "Missing data elements:",
              "i" = "Either `data` and/or `formula` or `x` and `y` must be supplied"
            )
          )
        }
        if (!(length(formula) == 3 && length(formula[[2]]) == 1) && is_formula(formula)) {
          abort(
            c(
              "Malformed formula:",
              "i" = "Please specify `formula` as a valid, two-sided formula"
            )
          )
        }
        return(list(formula = formula, data = data))
      } else if (any(vapply(list(x, y), is.null, NA))) {
        abort(
          c(
            "Missing data elements:",
            "i" = "Either `data` and/or `formula` or `x` and `y` must be supplied"
          )
        )
      }
      list(x = x, y = y)
    },
    convert_predictions = NULL,
    evaluation_data = NULL,
    fit_grid = function(input, response_var) {
      pb <- progressor(along = 1:nrow(self$tune_params))
      model_contents <- future_lapply(
        1:nrow(self$tune_params),
        function(grid_idx) {
          if ("x" %in% names(input)) {
            input <- input[c("x", "y")]
          } else {
            input <- input[c("formula", "data")]
          }
          parameters <- as.list(self$tune_params[grid_idx, ])
          fit <- eval_tidy(
            call2(
              self$learner,
              !!!input,
              !!!parameters,
              !!!private$learner_args
            )
          )
          preds <- lapply(
            private$prediction_args,
            function(.x) {
              eval_tidy(
                call2(
                  expr(predict),
                  fit,
                  private$evaluation_data$x,
                  !!!.x
                )
              )
            }
          )
          metrics <- Map(
            f = function(.x, .y, .z) {
              if (!is.null(.y)) {
                preds <- .y(unlist(.z))
              } else {
                preds <- unlist(.z)
              }
              eval_tidy(call_modify(.x, truth = response_var, estimate = preds))
            },
            self$scorer,
            if (is.null(private$convert_predictions)) list(NULL) else private$convert_predictions,
            preds
          )
          pb()
          list(model = fit, preds = preds, metrics = metrics)
        },
        future.seed = TRUE
      )
      models <- lapply(model_contents, function(i) i$model)
      preds <- lapply(model_contents, function(i) i$preds)
      metrics <- unlist(lapply(model_contents, function(i) i$metrics))
      metrics <- setNames(
        lapply(
          unique(names(metrics)),
          function(i) unname(metrics[names(metrics) == i])
        ),
        unique(names(metrics))
      )
      list(models = models, preds = preds, metrics = metrics)
    },
    learner_args = NULL,
    optimize_score = NULL,
    prediction_args = NULL,
    response = function(response_var) {
      if (is.factor(response_var)) {
        return(response_var)
      } else if (all(response_var %in% c(0, 1)) || is.logical(response_var)) {
        return(factor(response_var))
      }
      response_var
    }
  )
)
