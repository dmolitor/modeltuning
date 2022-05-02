#' Predictive Models with Cross Validation
#'
#' @description
#' `CV` allows the user to specify a cross validation scheme with complete
#' flexibility in the model, data splitting function, and performance metrics,
#' among other essential parameters.
#'
#' @export
CV <- R6Class(
  classname = "CV",
  public = list(
    #' @description
    #' `fit` performs cross validation with user-specified parameters.
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
    #' @return An object of class [FittedCV].
    #' @examples
    #' if (require(rpart) && require(rsample) && require(yardstick)) {
    #'
    #'   iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
    #'   iris_new$Species <- factor(iris_new$Species == "virginica")
    #'
    #'   ### Basic Example
    #'
    #'   iris_cv <- CV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     splitter = rsample::vfold_cv,
    #'     splitter_args = list(v = 3),
    #'     scorer = list(
    #'       "accuracy" = yardstick::accuracy_vec
    #'     ),
    #'     prediction_args = list(type = "class")
    #'   )
    #'   iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)
    #'
    #'   ### Example with multiple metric functions
    #'
    #'   iris_cv <- CV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     splitter = rsample::vfold_cv,
    #'     splitter_args = list(v = 3),
    #'     scorer = list(
    #'       "f_meas" = yardstick::f_meas_vec,
    #'       "accuracy" = yardstick::accuracy_vec,
    #'       "roc_auc" = yardstick::roc_auc_vec,
    #'       "pr_auc" = yardstick::pr_auc_vec
    #'     ),
    #'     prediction_args = list(
    #'       "f_meas" = list(type = "class"),
    #'       "accuracy" = list(type = "class"),
    #'       "roc_auc" = list(type = "prob"),
    #'       "pr_auc" = list(type = "prob")
    #'     ),
    #'     convert_predictions = list(
    #'       NULL,
    #'       NULL,
    #'       function(i) i[, "FALSE"],
    #'       function(i) i[, "FALSE"]
    #'     )
    #'   )
    #'   iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)
    #' }
    fit = function(formula = NULL, data = NULL, x = NULL, y = NULL, progress = FALSE) {
      input <- private$check_data_input(formula, data, x, y)
      n_obs <- if (!is.null(x)) nrow(x) else nrow(data)
      cv_index <- private$split_data(input, self$splitter)
      response_var <- private$response(input)
      nfolds <- length(cv_index)
      if (progress) {
        with_progress({
          model_output <- private$fit_folds(cv_index, input, response_var, n_obs)
        })
      } else {
        model_output <- private$fit_folds(cv_index, input, response_var, n_obs)
      }
      FittedCV$new(
        folds = cv_index,
        model = model_output$model,
        metrics = model_output$metrics,
        nfolds = nfolds,
        predictions = model_output$preds
      )
    },
    #' @description
    #' Create a new [CV] object.
    #'
    #' @param learner Function that estimates a predictive model. It is
    #'   essential that this function support either a formula interface with
    #'   `formula` and `data` arguments, or an alternate matrix interface with
    #'   `x` and `y` arguments.
    #' @param splitter A function that computes cross validation folds from an
    #'   input data set or a pre-computed list of cross validation fold indices.
    #'   If `splitter` is a function, it must have a `data` argument for the
    #'   input data, and it must return a list of cross validation fold indices.
    #'   If `splitter` is a list of integers, the number of cross validation
    #'   folds is `length(splitter)` and each element contains the indices of
    #'   the data observations that are included in that fold.
    #' @param scorer A named list of metric functions to evaluate model
    #'   performance on each cross validation fold. Any provided metric function
    #'   must have `truth` and `estimate` arguments, for true outcome values and
    #'   predicted outcome values respectively, and must return a single numeric
    #'   metric value.
    #' @param learner_args A named list of additional arguments to pass to
    #'   `learner`.
    #' @param splitter_args A named list of additional arguments to pass to
    #'   `splitter`.
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
    #' @return An object of class [CV].
    initialize = function(learner = NULL,
                          splitter = NULL,
                          scorer = NULL,
                          learner_args = NULL,
                          splitter_args = NULL,
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
      if (is.null(splitter)){
        abort(
          c(
            "Missing argument:",
            "x" = "`splitter` must be specified"
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
      self$learner <- enexpr(learner)
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(learner)
      )
      private$learner_args <- learner_args
      self$splitter <- if (is.list(splitter)) {
        splitter
      } else {
        call2(enexpr(splitter), !!!splitter_args)
      }
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(splitter)
      )
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
    },
    #' @field learner Predictive modeling function.
    learner = NULL,
    #' @field scorer List of performance metric functions.
    scorer = NULL,
    #' @field splitter Function that splits data into cross validation folds.
    splitter = NULL
  ),
  private = list(
    # Validate data inputs
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
    # Convert predicted values into acceptable scoring format
    convert_predictions = NULL,
    # Fit cross-validated model
    fit_folds = function(cv_index, input, response_var, n_obs) {
      pb <- progressor(along = 1:(length(cv_index) + 1))
      model_contents <- future_lapply(
        append(cv_index, list(1:n_obs)),
        function(idx) {
          data_in <- private$subset_data(
            formula = input$formula,
            data = input$data,
            x = input$x,
            y = input$y,
            idx = idx
          )
          if ("x" %in% names(data_in)) {
            data_out <- data_in[["x_out"]]
            data_in <- data_in[c("x", "y")]
          } else {
            data_out <- data_in[["data_out"]]
            data_in <- data_in[c("formula", "data")]
          }
          fit <- eval_tidy(call2(self$learner, !!!data_in, !!!private$learner_args))
          if (nrow(data_out) == 0) {
            preds <- list(NULL)
            metrics <- list(NULL)
          } else {
            preds <- lapply(
              private$prediction_args,
              function(.x) {
                eval_tidy(
                  call2(
                    expr(predict),
                    fit,
                    data_out,
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
                truth <- response_var[-idx]
                eval_tidy(call_modify(.x, truth = truth, estimate = preds))
              },
              self$scorer,
              if (is.null(private$convert_predictions)) list(NULL) else private$convert_predictions,
              preds
            )
          }
          pb()
          list(model = fit, preds = preds, metrics = metrics)
        },
        future.globals = structure(TRUE, add = modelselection_fns()),
        future.packages = private$future_packages,
        future.seed = TRUE
      )
      model <- model_contents[[length(model_contents)]]$model
      preds <- lapply(
        model_contents[1:(length(model_contents) - 1)],
        function(i) i$preds
      )
      metrics <- unlist(
        lapply(
          model_contents[1:(length(model_contents) - 1)],
          function(i) i$metrics
        )
      )
      metrics <- setNames(
        lapply(
          unique(names(metrics)),
          function(i) unname(metrics[names(metrics) == i])
        ),
        unique(names(metrics))
      )
      list(model = model, preds = preds, metrics = metrics)
    },
    # Get all packages required to evaluate futures
    future_packages = c("future.apply", "progressr", "R6", "rlang"),
    # Arguments to pass to learner function
    learner_args = NULL,
    # Arguments to pass to prediction method
    prediction_args = NULL,
    # Extracts response variable
    response = function(data_list) {
      if (!is.null(data_list[["formula"]])) {
        lhs <- as_string(data_list[["formula"]][[2]])
        response_var <- data_list[["data"]][, lhs]
      } else {
        response_var <- data_list[["y"]]
      }
      if (is.factor(response_var)) {
        return(response_var)
      } else if (all(response_var %in% c(0, 1)) || is.logical(response_var)) {
        return(factor(response_var))
      }
      response_var
    },
    # Splitter for input data (either a function or a list with fold indices)
    split_data = function(data_list, splitter) {
      if (is.list(splitter)) {
        return(splitter)
      }
      split_call <- if ("x" %in% names(data_list)) {
        call_modify(splitter, data = data_list[["x"]])
      } else {
        call_modify(splitter, data = data_list[["data"]])
      }
      data_splits <- eval_tidy(split_call)
      lapply(data_splits$splits, function(i) i$in_id)
    },
    # Subsets data for cross-validation folds
    subset_data = function(formula, data, x, y, idx) {
      if (is.null(data)) {
        return(
          list(
            x = x[idx, , drop = FALSE],
            y = y[idx],
            x_out = x[-idx, , drop = FALSE]
          )
        )
      } else {
        return(
          list(
            formula = formula,
            data = data[idx, , drop = FALSE],
            data_out = data[-idx, , drop = FALSE]
          )
        )
      }
    }
  )
)

#' Fitted, Cross-Validated Predictive Models
#'
#' @description
#' `FittedCV` is a fitted, cross-validated predictive model object that is
#' returned by `CV$fit()` and contains relevant model components,
#' cross-validation metrics, validation set predicted values, etc.
#'
#' @export
FittedCV <- R6Class(
  classname = "FittedCV",
  public = list(
    #' @field folds A list of length `$nfolds` where each element contains the
    #'   indices of the observations contained in that fold.
    folds = NULL,
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
    initialize = function(folds, model, metrics, nfolds, predictions) {
      self$folds <- folds
      self$model <- model
      self$mean_metrics <- lapply(metrics, mean)
      self$metrics <- metrics
      self$nfolds <- nfolds
      self$predictions <- predictions
    },
    #' @field model Predictive model fitted on the full data set.
    model = NULL,
    #' @field mean_metrics Numeric list; Cross-validation performance metrics
    #'   averaged across folds.
    mean_metrics = NULL,
    #' @field metrics Numeric list; Cross-validation performance metrics on each
    #'   fold.
    metrics = NULL,
    #' @field nfolds An integer specifying the number of cross-validation folds.
    nfolds = NULL,
    #' @field predictions A list containing the predicted hold-out values on
    #'   every fold.
    predictions = NULL
  )
)
