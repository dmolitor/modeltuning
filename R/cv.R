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
    #' @param response String; In the absence of `formula` or `y`, this specifies
    #'   which element of `learner_args` is the response vector.
    #' @param convert_response Function; This should be a single function that
    #'   transforms the response vector. E.g. a function converting a numeric binary
    #'   variable to a factor variable.
    #' @param progress Logical; indicating whether to print progress across
    #'   cross validation folds.
    #' @return An object of class [FittedCV].
    #' @examples
    #' if (require(e1071) && require(rpart) && require(yardstick)) {
    #'   iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
    #'   iris_new$Species <- factor(iris_new$Species == "virginica")
    #'
    #'   ### Decision Tree Example
    #'
    #'   iris_cv <- CV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     splitter = cv_split,
    #'     scorer = list(accuracy = yardstick::accuracy_vec),
    #'     prediction_args = list(accuracy = list(type = "class"))
    #'   )
    #'   iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)
    #'
    #'   ### Example with multiple metric functions
    #'
    #'   iris_cv <- CV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 3),
    #'     scorer = list(
    #'       f_meas = yardstick::f_meas_vec,
    #'       accuracy = yardstick::accuracy_vec,
    #'       roc_auc = yardstick::roc_auc_vec,
    #'       pr_auc = yardstick::pr_auc_vec
    #'     ),
    #'     prediction_args = list(
    #'       f_meas = list(type = "class"),
    #'       accuracy = list(type = "class"),
    #'       roc_auc = list(type = "prob"),
    #'       pr_auc = list(type = "prob")
    #'     ),
    #'     convert_predictions = list(
    #'       f_meas = NULL,
    #'       accuracy = NULL,
    #'       roc_auc = function(i) i[, "FALSE"],
    #'       pr_auc = function(i) i[, "FALSE"]
    #'     )
    #'   )
    #'   iris_cv_fitted <- iris_cv$fit(formula = Species ~ ., data = iris_new)
    #'
    #'   # Print the mean performance metrics across CV folds
    #'   iris_cv_fitted$mean_metrics
    #'
    #'   # Grab the final model fitted on the full dataset
    #'   iris_cv_fitted$model
    #'
    #'   ### OLS Example
    #'
    #'   mtcars_cv <- CV$new(
    #'     learner = lm,
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 2),
    #'     scorer = list("rmse" = yardstick::rmse_vec, "mae" = yardstick::mae_vec)
    #'   )
    #'
    #'   mtcars_cv_fitted <- mtcars_cv$fit(
    #'     formula = mpg ~ .,
    #'     data = mtcars
    #'   )
    #'
    #'   ### Matrix interface example - SVM
    #'
    #'   mtcars_x <- model.matrix(mpg ~ . - 1, mtcars)
    #'   mtcars_y <- mtcars$mpg
    #'
    #'   mtcars_cv <- CV$new(
    #'     learner = e1071::svm,
    #'     learner_args = list(scale = TRUE, kernel = "polynomial", cross = 0),
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 3),
    #'     scorer = list(rmse = yardstick::rmse_vec, mae = yardstick::mae_vec)
    #'   )
    #'   mtcars_cv_fitted <- mtcars_cv$fit(
    #'     x = mtcars_x,
    #'     y = mtcars_y
    #'   )
    #' }
    fit = function(formula = NULL,
                   data = NULL,
                   x = NULL,
                   y = NULL,
                   response = NULL,
                   convert_response = NULL,
                   progress = FALSE) {
      input <- private$check_data_input(formula, data, x, y, response, convert_response)
      n_obs <- if (!is.null(x)) nrow(x) else nrow(data)
      cv_index <- private$split_data(input, self$splitter)
      nfolds <- length(cv_index)
      if (progress) {
        with_progress({
          model_output <- private$fit_folds(cv_index, input, n_obs, response, convert_response)
        })
      } else {
        model_output <- private$fit_folds(cv_index, input, n_obs, response, convert_response)
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
    #'   must have `truth` and `estimate` arguments for true outcome values and
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
      # Argument checking
      check_list_or_null(
        learner_args = enexpr(learner_args),
        splitter_args = enexpr(splitter_args),
        scorer_args = enexpr(scorer_args),
        prediction_args = enexpr(prediction_args),
        convert_predictions = convert_predictions
      )
      if (is.null(enexpr(learner))){
        abort(c("Missing argument:", "x" = "`learner` must be specified"))
      }
      if (is.null(splitter)){
        abort(c("Missing argument:", "x" = "`splitter` must be specified"))
      }
      validate_scorer(scorer)
      validate_splitter(splitter)
      compare_names(scorer = scorer, convert_predictions = convert_predictions)
      # Nicely check scorer_args and prediction_args without evaluation happening
      scorer_args_nse <- expr_to_quoted_list(enexpr(scorer_args))
      prediction_args_nse <- expr_to_quoted_list(enexpr(prediction_args))
      compare_names(scorer = scorer, scorer_args = scorer_args_nse)
      compare_names(scorer = scorer, prediction_args = prediction_args_nse)

      # Initialize attributes
      self$learner <- enexpr(learner)
      private$learner_args <- enexpr(learner_args)
      self$splitter <- if (is.list(splitter)) {
        splitter
      } else {
        call2(enexpr(splitter), !!!expr_to_quoted_list(enexpr(splitter_args)))
      }
      self$scorer <- scorer
      private$scorer_args <- if (is.null(enexpr(scorer_args))) {
        expr(list(NULL))
      } else {
        enexpr(scorer_args)
      }
      private$prediction_args <- if (is.null(enexpr(prediction_args))) {
        expr(list(NULL))
      } else {
        enexpr(prediction_args)
      }
      private$convert_predictions <- if (
        !is.null(convert_predictions) &&
        !rlang::is_list(convert_predictions)
      ) {
        if (!is.function(convert_predictions)) abort("`convert_predictions` should be a function")
        list(convert_predictions)
      } else {
        convert_predictions
      }

      # Append packages needed for future to process in parallel
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(eval(self$learner))
      )
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(splitter)
      )
      private$future_packages <- append(
        private$future_packages,
        unname(lapply(scorer, get_namespace_name))
      )
      private$future_packages <- append(
        private$future_packages,
        unname(lapply(private$convert_predictions, get_namespace_name))
      )
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
    check_data_input = function(formula = NULL,
                                data = NULL,
                                x = NULL,
                                y = NULL,
                                response = NULL,
                                convert_response = NULL) {
      # Make sure the user provides something
      if (all(vapply(list(formula, data, x, y, response), is.null, NA))) {
        abort(c(
          "Missing data elements:",
          "x" = "No data elements were provided",
          "i" = "Provide `data` (and possibly `formula`), `x` (and possibly `y`, or `response`"
        ))
      }
      # A response variable must be specified
      if (is.null(formula) && is.null(y) && is.null(response)) {
        rlang::abort("One of `formula`, `y`, or `response` must be provided")
      }
      # Make sure they provide at least one of x or data
      if (is.null(data) && is.null(x)) {
        abort(c(
          "Missing elements:",
          "i" = "One of `data` or `x` should always be provided"
        ))
      }
      # Check if they're correctly using the data/formula interface
      if (!is.null(formula)) {
        if (is.null(data)) {
          abort(c(
            "Missing elements:",
            "i" = "if `formula` was supplied, `data` should usually be as well"
          ))
        }
      }
      # Checks for outcome specification
      if (!is.null(response) && (!is.null(formula) || !is.null(y))) {
        rlang::abort("If `formula` or `y` are provided `response` should not be")
      }
      if (!is.null(response) && !(is.character(response) || is.function(response))) {
        rlang::abort("`response` must be NULL, a string, or a function")
      }
      if (!is.null(convert_response) && !is.function(convert_response)) {
        rlang::abort("`convert_response` should either be NULL or a function")
      }
      # Return any non-null learner args
      learner_args <- list(x = x, y = y, formula = formula, data = data)
      is_null <- vapply(learner_args, function(.x) is.null(.x), NA)
      learner_args <- learner_args[!is_null]
      learner_args
    },

    # Convert predicted values into acceptable scoring format
    convert_predictions = NULL,

    # Fit cross-validated model
    fit_folds = function(cv_index, input, n_obs, response, convert_response) {
      pb <- progressor(along = 1:(length(cv_index) + 1))
      model_contents <- future_lapply(
        append(cv_index, list(1:n_obs)),
        function(idx) {
          data <- private$subset_data(
            formula = input$formula,
            data = input$data,
            x = input$x,
            y = input$y,
            idx = idx
          )
          data_out <- data[["data_out"]]
          truth <- data[["response_out"]]
          data_in <- data[!names(data) %in% c("data_out", "response_out")]
          if ("x" %in% names(data_in)) {
            # Evaluate learner arguments with data masking
            learner_args <- eval_tidy(
              expr = private$learner_args,
              env = rlang::env(
                rlang::caller_env(),
                .data = data_in[["x"]],
                .index = idx
              )
            )
          } else {
            # Evaluate learner arguments with data masking
            learner_args <- eval_tidy(
              private$learner_args,
              env = rlang::env(
                rlang::caller_env(),
                .data = data_in[["data"]],
                .index = idx
              )
            )
          }
          # Get the response (outcome) vector for both the in- and out-of- sample
          # data. Then apply any post-processing function, if supplied by the user.
          if (!is.null(response)) {
            if (is.character(response)) {
              # Pull out the response vector for the hold-out set
              truth <- learner_args[[response]][-idx]
              # Pull out the in-sample response vector
              learner_args[[response]] <- learner_args[[response]][idx]
            } else if (is.function(response)) {
              truth <- response(data_out)
            }
          }
          # Apply any user-supplied post-processing to the out-of-sample outcomes
          if (!is.null(convert_response) && length(truth) > 0) {
            truth <- convert_response(truth)
          }
          fit <- eval_tidy(call2(self$learner, !!!data_in, !!!learner_args))
          if (nrow(data_out) == 0) {
            # When training model on full data-set generate no fitted values
            # or model evaluation metrics
            preds <- list(NULL)
            metrics <- list(NULL)
          } else {
            # Construct scorer functions
            scorer_args <- eval_tidy(
              private$scorer_args,
              env = rlang::env(
                rlang::caller_env(),
                .data = data_out,
                .index = -idx
              )
            )
            scorer <- Map(f = function(.x, .y) call2(.x, !!!.y), self$scorer, scorer_args)
            # Evaluate prediction arguments with data masking
            prediction_args <- eval_tidy(
              private$prediction_args,
              env = rlang::env(
                rlang::caller_env(),
                .data = data_out,
                .index = -idx
              )
            )
            # Generate fitted values on hold-out set
            preds <- lapply(
              prediction_args,
              function(.x) { eval_tidy(call2(expr(predict), fit, data_out, !!!.x)) }
            )
            # Generate model metrics on hold-out set
            metrics <- Map(
              f = function(.x, .y, .z) {
                if (!is.null(.y)) {
                  preds <- .y(.z)
                } else {
                  if (is.recursive(.z)) {
                    abort(
                      c(
                        "Mis-shaped Predictions:",
                        "x" = "Predicted values should be an atomic vector",
                        "i" = paste(
                          "You may need to specify `convert_predictions` to",
                          "transform the output of `predict` appropriately"
                        )
                      ),
                      call = NULL
                    )
                  }
                  preds <- .z
                }
                eval_tidy(call_modify(.x, truth = truth, estimate = preds))
              },
              scorer,
              if (is.null(private$convert_predictions)) list(NULL) else private$convert_predictions,
              preds
            )
          }
          pb()
          list(model = fit, preds = preds, metrics = metrics)
        },
        future.globals = structure(
          TRUE,
          add = append(
            modelselection_fns(),
            c(
              get_objects_from_env(extract_symbols(self$learner)),
              get_objects_from_env(extract_symbols(private$learner_args)),
              get_objects_from_env(extract_symbols(private$scorer_args)),
              get_objects_from_env(extract_symbols(private$prediction_args)),
              get_objects_from_env(extract_symbols(self$splitter))
            )
          )
        ),
        future.packages = unique(append(private$future_packages, attached_pkgs())),
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
      metrics <- set_names(
        lapply(
          unique(names(metrics)),
          function(i) unlist(unname(metrics[names(metrics) == i]))
        ),
        unique(names(metrics))
      )
      list(model = model, preds = preds, metrics = metrics)
            },

    # Get all packages required to evaluate futures
    future_packages = c("future.apply", "progressr", "R6", "rlang", "Matrix"),

    # Arguments to pass to learner function
    learner_args = NULL,

    # Arguments to pass to prediction method
    prediction_args = NULL,

    # Extracts response variable
    response = function(data_list) {
      if (!is.null(data_list[["formula"]])) {
        lhs <- as_string(data_list[["formula"]][[2]])
        response_var <- data_list[["data"]][, lhs, drop = TRUE]
      } else if (!is.null(data_list[["y"]])) {
        response_var <- data_list[["y"]]
      } else {
        response_var <- NULL
      }
      # NOTE: Currently this is modifying the underlying outcome variable.
      # Again, I think this is probably dangerous! We should assume that
      # the user has correctly encoded the response variable and not
      # do anything weird to it. Again noting, in case I need to revert.
      return(response_var)
    },

    # Arguments to pass to scorer functions
    scorer_args = NULL,

    # Splitter for input data (either a function or a list with fold indices)
    split_data = function(data_list, splitter) {
      if (is.list(splitter)) {
        return(splitter)
      }
      if ("x" %in% names(data_list)) {
        eval_data <- data_list[["x"]]
      } else {
        eval_data <- data_list[["data"]]
      }
      split_call <- call_modify(splitter, data = eval_data)
      data_splits <- eval_tidy(
        split_call,
        env = rlang::env(rlang::caller_env(), .data = eval_data)
      )
      return(data_splits)
    },

    # Subsets data for cross-validation folds
    subset_data = function(formula, data, x, y, idx) {
      response_var <- private$response(list(
        formula = formula,
        data = data,
        x = x,
        y = y
      ))
      model_elts <- list()
      model_elts$formula <- formula
      model_elts$data <- data[idx, , drop = FALSE]
      model_elts$x <- x[idx, , drop = FALSE]
      model_elts$y <- y[idx]
      if (is.null(x)) {
        model_elts$data_out <- data[-idx, , drop = FALSE]
      } else {
        model_elts$data_out <- x[-idx, , drop = FALSE]
      }
      model_elts$response_out <- response_var[-idx]
      return(model_elts)
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
