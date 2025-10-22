#' Tune Predictive Model Hyper-parameters with Grid Search and Cross-Validation
#'
#' @description
#' `GridSearchCV` allows the user to specify a Grid Search schema for tuning
#' predictive model hyper-parameters with Cross-Validation. `GridSearchCV` gives
#' the user complete flexibility in the predictive model and performance
#' metrics.
#'
#' @export
GridSearchCV <- R6Class(
  classname = "GridSearchCV",
  public = list(

    #' @description
    #' `fit` tunes user-specified model hyper-parameters via Grid Search and
    #' Cross-Validation.
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
    #'   the hyper-parameter grid.
    #' @return An object of class [FittedGridSearchCV].
    #' @examples
    #' \donttest{
    #' if (require(e1071) && require(rpart) && require(yardstick)) {
    #'   iris_new <- iris[sample(1:nrow(iris), nrow(iris)), ]
    #'   iris_new$Species <- factor(iris_new$Species == "virginica")
    #'   iris_train <- iris_new[1:100, ]
    #'   iris_validate <- iris_new[101:150, ]
    #'
    #'   ### Decision Tree example
    #'
    #'   iris_grid_cv <- GridSearchCV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     tune_params = list(
    #'       minsplit = seq(10, 30, by = 5),
    #'       maxdepth = seq(20, 30, by = 2)
    #'     ),
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 3),
    #'     scorer = list(accuracy = yardstick::accuracy_vec),
    #'     optimize_score = "max",
    #'     prediction_args = list(accuracy = list(type = "class"))
    #'   )
    #'   iris_grid_cv_fitted <- iris_grid_cv$fit(
    #'     formula = Species ~ .,
    #'     data = iris_train
    #'   )
    #'
    #'   ### Example with multiple metric functions
    #'
    #'   iris_grid_cv <- GridSearchCV$new(
    #'     learner = rpart::rpart,
    #'     learner_args = list(method = "class"),
    #'     tune_params = list(
    #'       minsplit = seq(10, 30, by = 5),
    #'       maxdepth = seq(20, 30, by = 2)
    #'     ),
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 3),
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
    #'   iris_grid_cv_fitted <- iris_grid_cv$fit(
    #'     formula = Species ~ .,
    #'     data = iris_train,
    #'   )
    #'
    #'   # Grab the best model
    #'   iris_grid_cv_fitted$best_model
    #'
    #'   # Grab the best hyper-parameters
    #'   iris_grid_cv_fitted$best_params
    #'
    #'   # Grab the best model performance metrics
    #'   iris_grid_cv_fitted$best_metric
    #'
    #'   ### Matrix interface example - SVM
    #'
    #'   mtcars_train <- mtcars[1:25, ]
    #'   mtcars_eval <- mtcars[26:nrow(mtcars), ]
    #'
    #'   mtcars_grid_cv <- GridSearchCV$new(
    #'     learner = e1071::svm,
    #'     tune_params = list(
    #'       degree = 2:4,
    #'       kernel = c("linear", "polynomial")
    #'     ),
    #'     splitter = cv_split,
    #'     splitter_args = list(v = 2),
    #'     learner_args = list(scale = TRUE),
    #'     scorer = list(
    #'       rmse = yardstick::rmse_vec,
    #'       mae = yardstick::mae_vec
    #'     ),
    #'     optimize_score = "min"
    #'   )
    #'   mtcars_grid_cv_fitted <- mtcars_grid_cv$fit(
    #'     x = mtcars_train[, -1],
    #'     y = mtcars_train$mpg
    #'   )
    #'
    #' }
    #' }
    fit = function(formula = NULL,
                   data = NULL,
                   x = NULL,
                   y = NULL,
                   response = NULL,
                   convert_response = NULL,
                   progress = FALSE) {
      input <- private$check_data_input(formula, data, x, y)
      if (progress) {
        with_progress({
          models_output <- private$fit_grid(input, response, convert_response)
        })
      } else {
        models_output <- private$fit_grid(input, response, convert_response)
      }
      FittedGridSearchCV$new(
        tune_params = self$tune_params,
        models = lapply(models_output, function(i) i$model),
        folds = lapply(models_output, function(i) i$folds),
        metrics = lapply(models_output, function(i) i$mean_metrics),
        predictions = lapply(models_output, function(i) i$predictions),
        optimize_score = private$optimize_score
      )
    },

    #' @description
    #' Create a new [GridSearchCV] object.
    #'
    #' @param learner Function that estimates a predictive model. It is
    #'   essential that this function support either a formula interface with
    #'   `formula` and `data` arguments, or an alternate matrix interface with
    #'   `x` and `y` arguments.
    #' @param tune_params A named list specifying the arguments of `$learner` to
    #'   tune.
    #' @param splitter A function that computes cross validation folds from an
    #'   input data set or a pre-computed list of cross validation fold indices.
    #'   If `splitter` is a function, it must have a `data` argument for the
    #'   input data, and it must return a list of cross validation fold indices.
    #'   If `splitter` is a list of integers, the number of cross validation
    #'   folds is `length(splitter)` and each element contains the indices of
    #'   the data observations that are included in that fold.
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
    #' @return An object of class [GridSearch].
    initialize = function(learner = NULL,
                          tune_params = NULL,
                          splitter = NULL,
                          scorer = NULL,
                          optimize_score = c("min", "max"),
                          learner_args = NULL,
                          splitter_args = NULL,
                          scorer_args = NULL,
                          prediction_args = NULL,
                          convert_predictions = NULL) {
      # Argument validation
      check_list_or_null(
        learner_args = enexpr(learner_args),
        splitter_args = enexpr(splitter_args),
        scorer_args = enexpr(scorer_args),
        prediction_args = enexpr(prediction_args),
        convert_predictions = convert_predictions
      )
      if (is.null(enexpr(learner))){
        abort(c( "Missing argument:", "x" = "`learner` must be specified"))
      }
      if (is.null(tune_params)) {
        abort(c("Missing argument:", "x" = "`tune_params` must be specified"))
      }
      if (is.null(names(tune_params)) || any(vapply(names(tune_params), function(i) i == "", NA))) {
        abort(
          c(
            "Missing attribute names:",
            "x" = "Each element of `tune_params` must have a name corresponding to an argument of `learner`"
          )
        )
      }
      validate_scorer(scorer)
      validate_splitter(splitter)
      compare_names(scorer = scorer, convert_predictions = convert_predictions)
      # Nicely check scorer_args and prediction_args without evaluation happening
      scorer_args_nse <- expr_to_quoted_list(enexpr(scorer_args))
      prediction_args_nse <- expr_to_quoted_list(enexpr(prediction_args))
      compare_names(scorer = scorer, scorer_args = scorer_args_nse)
      compare_names(scorer = scorer, prediction_args = prediction_args_nse)

      # Initialize attributes and methods
      self$learner <- enexpr(learner)
      private$learner_args <- enexpr(learner_args)
      self$splitter <- splitter
      private$splitter_args <- enexpr(splitter_args)
      self$scorer <- scorer
      private$scorer_args <- if (is.null(enexpr(scorer_args))) {
        rlang::expr(list(NULL))
      } else {
        rlang::enexpr(scorer_args)
      }
      private$prediction_args <- if (is.null(enexpr(prediction_args))) {
        rlang::expr(list(NULL))
      } else {
        rlang::enexpr(prediction_args)
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
      self$tune_params <- expand.grid(tune_params, stringsAsFactors = FALSE)
      private$optimize_score <- match.arg(optimize_score)

      # Collect packages required for future parallelization
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(splitter)
      )
      private$future_packages <- append(
        private$future_packages,
        get_namespace_name(learner)
      )
      private$future_packages <- append(
        private$future_packages,
        unname(lapply(private$convert_predictions, get_namespace_name))
      )
      private$future_packages <- append(
        private$future_packages,
        unname(lapply(scorer, get_namespace_name))
      )
      private$future_packages <- sort(unlist(private$future_packages))
    },

    #' @field learner Predictive modeling function.
    learner = NULL,

    #' @field scorer List of performance metric functions.
    scorer = NULL,

    #' @field splitter Function that splits data into cross validation folds.
    splitter = NULL,

    #' @field tune_params Data.frame of full hyper-parameter grid created from
    #'   `$tune_params`
    tune_params = NULL

  ),
  private = list(

    # Validate data inputs
    check_data_input = function(formula = NULL, data = NULL, x = NULL, y = NULL) {
      # Make sure the user provides something
      if (all(vapply(list(formula, data, x, y), is.null, NA))) {
        abort(c(
          "Missing data elements:",
          "x" = "No data elements were provided",
          "i" = "Provide `data` (and possibly `formula`) or `x` (and possibly `y`)"
        ))
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
      # Return any non-null learner args
      learner_args <- list(x = x, y = y, formula = formula, data = data)
      is_null <- vapply(learner_args, function(.x) is.null(.x), NA)
      learner_args <- learner_args[!is_null]
      learner_args
    },

    # Convert predicted values into acceptable scoring format
    convert_predictions = NULL,

    # Fit CV models across hyper-parameter grid
    fit_grid = function(input, response, convert_response) {
      pb <- progressor(along = 1:nrow(self$tune_params))
      model_contents <- future_lapply(
        1:nrow(self$tune_params),
        function(grid_idx) {
          parameters <- extract_params(self$tune_params, index = grid_idx)
          # Append learner args without evaluating them
          learner_args <- expr_to_quoted_list(private$learner_args)
          parameters <- rlang::call2(expr(list), !!!append(parameters, learner_args))
          cv_model <- CV$new(
            learner = !!self$learner,
            splitter = self$splitter,
            scorer = self$scorer,
            learner_args = !!parameters,
            splitter_args = !!private$splitter_args,
            scorer_args = !!private$scorer_args,
            prediction_args = !!private$prediction_args,
            convert_predictions = private$convert_predictions
          )
          input$progress <- FALSE
          input$response <- response
          input$convert_response <- convert_response
          cv_model_call <- rlang::call2(expr(cv_model$fit), !!!input)
          cv_model <- rlang::eval_tidy(cv_model_call)
          pb()
          cv_model
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
              get_objects_from_env(extract_symbols(self$splitter)),
              get_objects_from_env(extract_symbols(private$splitter_args))
            )
          )
        ),
        future.packages = unique(append(private$future_packages, attached_pkgs())),
        future.seed = TRUE
      )
      model_contents
    },

    # Get all packages required to evaluate futures
    future_packages = c("future.apply", "progressr", "R6", "rlang", "Matrix"),

    # Arguments to pass to learner function
    learner_args = NULL,

    # How to optimize CV score
    optimize_score = NULL,

    # Arguments to pass to prediction method
    prediction_args = NULL,

    # Arguments to pass to scorer function
    scorer_args = NULL,

    # Arguments to pass to splitter function
    splitter_args = NULL
  )
)

#' Fitted Models with Cross Validation across a Tuning Grid of Hyper-parameters
#'
#' @description
#' `FittedGridSearchCV` is an object containing fitted predictive models across
#' a tuning grid of hyper-parameters returned by `GridSearchCV$fit()` as well as
#' relevant model information such as the best performing model, best
#' hyper-parameters, etc.
#'
#' @export
FittedGridSearchCV <- R6Class(
  classname = "FittedGridSearchCV",
  public = list(
    
    #' @field best_idx An integer specifying the index of `$models` that
    #'   contains the best-performing model.
    best_idx = NULL,

    #' @field best_metric The average performance metric of the best model
    #'   across cross-validation folds.
    best_metric = NULL,

    #' @field best_model The best performing predictive model.
    best_model = NULL,

    #' @field best_params A named list of the hyper-parameters that result in
    #'   the optimal predictive model.
    best_params = NULL,

    #' @field folds A list of length `$models` where each element contains a
    #'   list of the cross-validation indices for each fold.
    folds = NULL,

    #' @field tune_params A [data.frame] of the full hyper-parameter grid.
    tune_params = NULL,

    #' @description
    #' Create a new [FittedGridSearchCV] object.
    #'
    #' @param tune_params Data.frame of the full hyper-parameter grid.
    #' @param models List of predictive models at every value of `$tune_params`.
    #' @param folds List of cross-validation indices at every value of
    #'   `$tune_params`.
    #' @param metrics List of cross-validation performance metrics for
    #'   every model in `$models`.
    #' @param predictions A list containing the predicted values on the
    #'   cross-validation folds for every model in `$models`.
    #' @param optimize_score Either "max" or "min" indicating whether or not the
    #'   specified performance metric was maximized or minimized to find the
    #'   optimal predictive model.
    #'
    #' @return An object of class [FittedGridSearchCV].
    initialize = function(tune_params,
                          models,
                          folds,
                          metrics,
                          predictions,
                          optimize_score) {
      self$tune_params <- tune_params
      self$models <- models
      self$folds <- folds
      self$predictions <- predictions
      metrics <- unlist(metrics)
      metrics <- set_names(
        lapply(
          unique(names(metrics)),
          function(i) unlist(unname(metrics[names(metrics) == i]))
        ),
        unique(names(metrics))
      )
      self$metrics <- metrics
      eval_metrics <- metrics[[length(metrics)]]
      self$best_idx <- eval_tidy(
        call2(
          paste0("which.", optimize_score),
          eval_metrics
        )
      )
      self$best_metric <- eval_metrics[[self$best_idx]]
      best_params <- as.list(tune_params[self$best_idx, , drop = FALSE])
      best_params <- lapply(best_params, function(.x) .x)
      self$best_params <- best_params
      self$best_model <- models[[self$best_idx]]
    },

    #' @field models List of predictive models at every value of `$tune_params`.
    models = NULL,

    #' @field metrics Numeric list; Cross-validation performance metrics for
    #'   every model in `$models`.
    metrics = NULL,

    #' @field predictions A list containing the cross-validation fold
    #'   predictions for each model in `$models`.
    predictions = NULL

  )
)
