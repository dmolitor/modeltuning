FittedGridSearchCV <- R6Class(
  classname = "FittedGridSearchCV",
  public = list(
    #' @field best_idx An integer specifying the index of `$models` that
    #'   contains the best-performing model.
    best_idx = NULL,
    #' @field best_metric The average performance metric of the best model across
    #'   cross-validation folds.
    best_metric = NULL,
    #' @field best_model The best performing predictive model.
    best_model = NULL,
    #' @field best_params A named list of the hyper-parameters that result in
    #'   the optimal predictive model.
    best_params = NULL,
    #' @field folds A list of length `$models` where each element contains a
    #'   list of the cross-validation indices for each fold.
    folds = NULL,
    #' @field tune_params Data.frame of the full hyper-parameter grid.
    tune_params = NULL,
    #' @description
    #' Create a new [FittedGridSearchCV] object.
    #'
    #' @param tune_params Data.frame of the full hyper-parameter grid.
    #' @param models List of predictive models at every value of `$tune_params`.
    #' @param metrics List of performance metrics on the validation data for
    #'   every model in `$models`.
    #' @param predictions A list containing the predicted values on the
    #'   cross-validation folds for every model in `$models`.
    #' @param optimize_score Either "max" or "min" indicating whether or not the
    #'   specified performance metric was maximized or minimized to find the
    #'   optimal predictive model.
    #'
    #' @return An object of class [FittedGridSearch].
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
      metrics <- setNames(
        lapply(
          unique(names(metrics)),
          function(i) unname(metrics[names(metrics) == i])
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
      self$best_params <- unlist(tune_params[self$best_idx, , drop = FALSE])
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

GridSearchCV <- R6Class(
  classname = "GridSearchCV",
  public = list(
    fit = function(formula = NULL,
                   data = NULL,
                   x = NULL,
                   y = NULL,
                   progress = FALSE) {
      input <- private$check_data_input(formula, data, x, y)
      if (progress) {
        with_progress({
          models_output <- private$fit_grid(input = input)
        })
      } else {
        models_output <- private$fit_grid(input = input)
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
      private$learner_args <- learner_args
      self$splitter <- if (is.list(splitter)) {
        splitter
      } else {
        enexpr(splitter)
      }
      private$splitter_args <- splitter_args
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
      self$scorer <- scorer
      private$scorer_args <- scorer_args
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
    # Fit CV models across hyper-parameter grid
    fit_grid = function(input) {
      pb <- progressor(along = 1:nrow(self$tune_params))
      model_contents <- future_lapply(
        1:nrow(self$tune_params),
        function(grid_idx) {
          parameters <- as.list(self$tune_params[grid_idx, , drop = FALSE])
          parameters <- append(parameters, private$learner_args)
          cv_model <- CV$new(
            learner = !!self$learner,
            splitter = self$splitter,
            scorer = self$scorer,
            learner_args = parameters,
            splitter_args = private$splitter_args,
            scorer_args = private$scorer_args,
            prediction_args = private$prediction_args,
            convert_predictions = private$convert_predictions
          )
          if ("x" %in% names(input)) {
            cv_model <- cv_model$fit(
              x = input$x,
              y = input$y,
              progress = FALSE
            )
          } else {
            cv_model <- cv_model$fit(
              formula = input$formula,
              data = input$data,
              progress = FALSE
            )
          }
          pb()
          cv_model
        },
        future.seed = TRUE
      )
      model_contents
    },
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
