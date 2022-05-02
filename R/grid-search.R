#' Fitted Models across a Tuning Grid of Hyper-parameters
#'
#' @description
#' `FittedGridSearch` is an object containing fitted predictive models across
#' a tuning grid of hyper-parameters returned by `GridSearch$fit()` as well as
#' relevant model information such as the best performing model, best
#' hyper-parameters, etc.
#'
#' @export
FittedGridSearch <- R6Class(
  classname = "FittedGridSearch",
  public = list(
    #' @field best_idx An integer specifying the index of `$models` that
    #'   contains the best-performing model.
    best_idx = NULL,
    #' @field best_metric The performance metric of the best model on the
    #'   validation data.
    best_metric = NULL,
    #' @field best_model The best performing predictive model.
    best_model = NULL,
    #' @field best_params A named list of the hyper-parameters that result in
    #'   the optimal predictive model.
    best_params = NULL,
    #' @field tune_params Data.frame of the full hyper-parameter grid.
    tune_params = NULL,
    #' @description
    #' Create a new [FittedGridSearch] object.
    #'
    #' @param tune_params Data.frame of the full hyper-parameter grid.
    #' @param models List of predictive models at every value of `$tune_params`.
    #' @param metrics List of performance metrics on the validation data for
    #'   every model in `$models`.
    #' @param predictions A list containing the predicted values on the
    #'   validation data for every model in `$models`.
    #' @param optimize_score Either "max" or "min" indicating whether or not the
    #'   specified performance metric was maximized or minimized to find the
    #'   optimal predictive model.
    #'
    #' @return An object of class [FittedGridSearch].
    initialize = function(tune_params,
                          models,
                          metrics,
                          predictions,
                          optimize_score) {
      self$tune_params <- tune_params
      self$models <- models
      self$metrics <- metrics
      self$predictions <- predictions
      eval_metrics <- metrics[[length(metrics)]]
      self$best_idx <- eval_tidy(
        call2(
          paste0("which.", optimize_score),
          eval_metrics
        )
      )
      self$best_metric <- eval_metrics[[self$best_idx]]
      self$best_params <- unlist(tune_params[self$best_idx, ])
      self$best_model <- models[[self$best_idx]]
    },
    #' @field models List of predictive models at every value of `$tune_params`.
    models = NULL,
    #' @field metrics Numeric list; Cross-validation performance metrics on each
    #'   fold.
    metrics = NULL,
    #' @field predictions A list containing the predicted hold-out values on
    #'   every fold.
    predictions = NULL
  )
)

#' Tune Predictive Model Hyper-parameters with Grid Search
#'
#' @description
#' `GridSearch` allows the user to specify a Grid Search schema for tuning
#' predictive model hyper-parameters with complete flexibility in the predictive
#' model and performance metrics.
#'
#' @export
GridSearch <- R6Class(
  classname = "GridSearch",
  public = list(
    #' @description
    #' `fit` tunes user-specified model hyper-parameters via Grid Search.
    #'
    #' @details
    #' `fit` follows standard R modeling convention by surfacing a formula
    #' modeling interface as well as an alternate matrix option. The user should
    #' use whichever interface is supported by the specified `$learner`
    #' function.
    #'
    #' @param formula An object of class [formula]: a symbolic description of
    #'   the model to be fitted.
    #' @param data An optional data frame, or other object containing the
    #'   variables in the model. If `data` is not provided, how `formula` is
    #'   handled depends on `$learner`.
    #' @param x Predictor data (independent variables), alternative interface to
    #'   data with formula.
    #' @param y Response vector (dependent variable), alternative interface to
    #'   data with formula.
    #' @param progress Logical; indicating whether to print progress across
    #'   cross validation folds.
    #' @return An object of class [FittedGridSearch].
    #' @examples
    #' if (require(rpart) && require(rsample) && require(yardstick)) {
    #'
    #'   iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
    #'   iris_new$Species <- factor(iris_new$Species == "virginica")
    #'   iris_train <- iris_new[1:100, ]
    #'   iris_validate <- iris_new[101:150, ]
    #'
    #'   ### Basic Example
    #'
    #'   iris_grid <- GridSearch$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     tune_params = list(
    #'       minsplit = seq(10, 30, by = 5),
    #'       maxdepth = seq(20, 30, by = 2)
    #'     ),
    #'     evaluation_data = list(x = iris_validate, y = iris_validate$Species),
    #'     scorer = list(accuracy = yardstick::accuracy_vec),
    #'     optimize_score = "max",
    #'     prediction_args = list(accuracy = list(type = "class"))
    #'   )
    #'   iris_grid_fitted <- iris_grid$fit(
    #'     formula = Species ~ .,
    #'     data = iris_train
    #'   )
    #'
    #'   ### Example with multiple metric functions
    #'
    #'   iris_grid <- GridSearch$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     tune_params = list(
    #'       minsplit = seq(10, 30, by = 5),
    #'       maxdepth = seq(20, 30, by = 2)
    #'     ),
    #'     evaluation_data = list(x = iris_validate, y = iris_validate$Species),
    #'     scorer = list(
    #'       accuracy = yardstick::accuracy_vec,
    #'       auc = yardstick::roc_auc_vec
    #'     ),
    #'     optimize_score = "max",
    #'     prediction_args = list(
    #'       accuracy = list(type = "class"),
    #'       auc = list(type = "prob")
    #'     ),
    #'     convert_predictions = list(
    #'       accuracy = NULL,
    #'       auc = function(i) i[, "FALSE"]
    #'     )
    #'   )
    #'   iris_grid_fitted <- iris_grid$fit(
    #'     formula = Species ~ .,
    #'     data = iris_train,
    #'   )
    #' }
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
        predictions = model_output$preds,
        optimize_score = private$optimize_score
      )
    },
    #' @description
    #' Create a new [GridSearch] object.
    #'
    #' @param learner Function that estimates a predictive model. It is
    #'   essential that this function support either a formula interface with
    #'   `formula` and `data` arguments, or an alternate matrix interface with
    #'   `x` and `y` arguments.
    #' @param tune_params A named list specifying the arguments of `$learner` to
    #'   tune.
    #' @param evaluation_data A two-element list containing the following
    #'   elements: `x`, the validation data to generate predicted values with;
    #'   `y`, the validation response values to evaluate predictive performance.
    #' @param scorer A named list of metric functions to evaluate model
    #'   performance on `evaluation_data`. Any provided metric function
    #'   must have `truth` and `estimate` arguments, for true outcome values and
    #'   predicted outcome values respectively, and must return a single numeric
    #'   metric value. The last metric function will be the one used to identify
    #'   the optimal model from the Grid Search.
    #' @param optimize_score One of "max" or "min"; Whether to maximize or
    #'   minimize the metric defined in `scorer` to find the optimal Grid Search
    #'   parameters.
    #' @param learner_args A named list of additional arguments to pass to
    #'   `learner`.
    #' @param scorer_args A named list of additional arguments to pass to
    #'   `scorer`. `scorer_args` must either be length 1 or `length(scorer)` in
    #'   the case where different arguments are being passed to each scoring
    #'   function.
    #' @param prediction_args A named list of additional arguments to pass to
    #'   `predict`. `prediction_args` must either be length 1 or
    #'   `length(scorer)` in the case where different arguments are being passed
    #'   to each scoring function.
    #' @param convert_predictions A list of functions to convert predicted
    #'   values prior to being evaluated by the metric functions supplied in
    #'   `scorer`. This list should either be length 1, in which case the same
    #'   function will be applied to all predicted values, or `length(scorer)`
    #'   in which case each function in `convert_predictions` will correspond
    #'   with each function in `scorer`.
    #'
    #' @return An object of class [GridSearch].
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
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(learner)
      )
      private$learner_args <- learner_args
      self$scorer <- Map(
        f = function(.x, .y) call2(.x, !!!.y),
        scorer,
        if (is.null(scorer_args)) list(NULL) else scorer_args
      )
      private$future_packages <- append(
        private$future_packages,
        unname(lapply(scorer, get_namespace_name))
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
        private$future_packages <- append(
          private$future_packages,
          get_namespace_name(convert_predictions)
        )
        list(convert_predictions)
      } else {
        private$future_packages <- append(
          private$future_packages,
          unname(lapply(convert_predictions, get_namespace_name))
        )
        convert_predictions
      }
      private$future_packages <- sort(unlist(private$future_packages))
      self$tune_params <- expand.grid(tune_params)
      private$optimize_score <- match.arg(optimize_score)
      private$evaluation_data <- evaluation_data
    },
    #' @field learner Predictive modeling function.
    learner = NULL,
    #' @field scorer List of performance metric functions.
    scorer = NULL,
    #' @field tune_params Data.frame of full hyper-parameter grid created from
    #'   `$tune_params`
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
          parameters <- as.list(self$tune_params[grid_idx, , drop = FALSE])
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
        future.globals = structure(TRUE, add = modelselection_fns()),
        future.packages = private$future_packages,
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
    future_packages = c("future.apply", "progressr", "R6", "rlang"),
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
