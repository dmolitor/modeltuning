# Scaling with AWS

This vignette will walk through an example of scaling a `modeltuning`
analysis with AWS via the [Paws SDK](https://github.com/paws-r/paws).
The following analysis depends on AWS credentials passed via environment
variables. To see a detailed outline of the different ways to set AWS
credentials, check out [this how-to
document](https://github.com/paws-r/paws/blob/main/docs/credentials.md).

## Requisite Packages

``` r

library(e1071)
library(future)
library(modeltuning) # devtools::install_github("dmolitor/modeltuning")
library(parallelly)
library(paws)
library(rsample)
library(yardstick)
```

## Data and Model Prep

### Data Prep

We’ll be training a classification model on the `iris` data-set to
predict whether a flower’s species is virginica or not.

First, let’s generate a bunch of synthetic data observations by adding
random noise to the original `iris` features and combining it into one
big dataframe.

``` r

iris_new <- do.call(
  what = rbind,
  args = replicate(n = 10, iris, simplify = FALSE)
) |>
  transform(
    Sepal.Length = jitter(Sepal.Length, 0.1),
    Sepal.Width = jitter(Sepal.Width, 0.1),
    Petal.Length = jitter(Petal.Length, 0.1),
    Petal.Width = jitter(Petal.Width, 0.1),
    Species = factor(Species == "virginica")
  )

# Shuffle the data-set
iris_new <- iris_new[sample(1:nrow(iris_new), nrow(iris_new)), ]

# Quick overview of the dataset
summary(iris_new[, 1:4])
#   Sepal.Length    Sepal.Width     Petal.Length     Petal.Width     
#  Min.   :4.299   Min.   :1.999   Min.   :0.9986   Min.   :0.09801  
#  1st Qu.:5.101   1st Qu.:2.799   1st Qu.:1.5984   1st Qu.:0.29972  
#  Median :5.799   Median :3.001   Median :4.3499   Median :1.30126  
#  Mean   :5.843   Mean   :3.057   Mean   :3.7580   Mean   :1.19938  
#  3rd Qu.:6.400   3rd Qu.:3.302   3rd Qu.:5.1002   3rd Qu.:1.80059  
#  Max.   :7.901   Max.   :4.401   Max.   :6.9020   Max.   :2.50193
```

### Grid Search Specification

Now that we’ve got the data prepped, let’s specify our predictive
modeling approach. For this analysis I’m going to train a Support Vector
classifier using the `e1071` package, and I’m going to use Grid Search
in combination with 5-fold Cross-Validation to find the optimal values
for the `cost` and `kernel` hyper-parameters.

``` r

# Create a splitter function that will return CV folds
splitter_fn <- function(data) lapply(vfold_cv(data, v = 5)$splits, \(y) y$in_id)

iris_grid <- GridSearchCV$new(
  learner = svm,
  tune_params = list(
    cost = c(0.01, 0.1, 0.5, 1, 3, 6),
    kernel = c("polynomial", "radial", "sigmoid")
  ),
  learner_args = list(
    scale = TRUE,
    type = "C-classification",
    probability = TRUE
  ),
  splitter = splitter_fn,
  scorer = list(
    accuracy = accuracy_vec,
    f_measure = f_meas_vec,
    auc = roc_auc_vec
  ),
  prediction_args = list(
    accuracy = NULL,
    f_measure = NULL,
    auc = list(probability = TRUE)
  ),
  convert_predictions = list(
    accuracy = NULL,
    f_measure = NULL,
    auc = function(.x) attr(.x, "probabilities")[, "FALSE"]
  ),
  optimize_score = "max"
)
```

Now that we’ve specified our Grid Search schema let’s check out the
hyper-parameter grid and see how many models we’re going to estimate.

``` r

cat("We will estimate", nrow(iris_grid$tune_params), "SVM models\n")
# We will estimate 18 SVM models
```

## Launch AWS Resources

To speed up the estimation of our models, let’s create a remote cluster
of 6 worker nodes to estimate the models in parallel.

### Launch EC2 Instances

First, we will launch 6 instances using a custom AMI that contains R and
a bunch of essential R packages. While this AMI is not available as a
community AMI there are definitely good AMIs out there that have a
comprehensive set of R packages and corresponding tools installed.
**Note:** which parameters you need to specify when launching EC2
instances may vary greatly depending on your account’s security
configurations.

``` r

ec2_client <- ec2()

# Request Instances
instance_req <- ec2_client$run_instances(
  ImageId = "ami-06dd49fc9e3a5acee",
  InstanceType = "t2.large",
  KeyName = key_name,
  MaxCount = 6,
  MinCount = 6,
  InstanceInitiatedShutdownBehavior = "terminate",
  SecurityGroupIds = security_group,
  # This names the instances
  TagSpecifications = list(
    list(
      ResourceType = "instance",
      Tags = list(
        list(
          Key = "Name",
          Value = "Worker Node"
        )
      )
    )
  )
)
```

Now that we’ve launched the instances we need to wait until they all
respond as `"running"` before we try to do anything (We also need to
wait for ~ 1 minute for the instances to initialize or they’ll reject
our SSH login attempts).

``` r

# Chalk up a quick function to return instance IDs from our request
instance_ids <- function(response) {
  vapply(response$Instances, function(i) i$InstanceId, character(1))
}

# Wait for instances to all respond as 'running'
while(
  !all(
    vapply(
      ec2_client$
      describe_instances(InstanceIds = instance_ids(instance_req))$
      Reservations[[1]]$
      Instances,
      function(i) i$State$Name,
      character(1)
    ) == "running"
  )
) {
  Sys.sleep(5)
}

# Rough heuristic -- give additional 45 seconds for instances to initialize
Sys.sleep(45)
```

### Create Cluster

Now, in order to set up our compute cluster we need to get the IP
addresses from these instances.

``` r

# Get public IPs
inst_public_ips <- vapply(
  ec2_client$
    describe_instances(InstanceIds = instance_ids(instance_req))$
    Reservations[[1]]$
    Instances,
  function(i) i$PublicIpAddress,
  character(1)
)
```

Finally, we can create a compute cluster on these worker nodes via SSH.

``` r

cl <- makeClusterPSOCK(
  worker = inst_public_ips,
  user = "ubuntu",
  rshopts = c("-o", "StrictHostKeyChecking=no",
              "-o", "IdentitiesOnly=yes",
              "-i", pem_fp), # Local filepath to private SSH key-pair
  connectTimeout = 25,
  tries = 3
)
```

## Estimate Models

Now that we’ve created our compute cluster, we can use the `future`
package to specify our parallelization plan. Since `modeltuning` is
built on top of the `future` framework, it will automatically
parallelize the model estimation across our 6-worker cluster. The
following parallelization **topology** basically is telling `future` to
parallelize the grid-search models across the compute cluster, and to
parallelize each model’s cross-validation across the cores of the
instance it is being evaluated on.

``` r

plan(
  list(
    tweak(cluster, workers = cl),
    multisession
  )
)
```

Finally, let’s estimate our Grid Search models in parallel!

``` r

iris_grid_fitted <- iris_grid$fit(
  formula = Species ~ .,
  data = iris_new,
  progress = TRUE
)
```

## Best Model/Parameters

Let’s check out the info on our best model.

``` r

best_idx <- iris_grid_fitted$best_idx
metrics <- iris_grid_fitted$metrics

# Print model metrics of best model
cat(
  " Accuracy:", round(100 * metrics$accuracy[[best_idx]], 2),
  "%\nF-Measure:", round(100 * metrics$f_measure[[best_idx]], 2),
  "%\n      AUC:", round(metrics$auc[[best_idx]], 4), "\n"
)
#  Accuracy: 98.4 %
# F-Measure: 98.81 %
#       AUC: 0.9993

params <- iris_grid_fitted$best_params

# Print the best hyper-parameters
cat(
  "  Optimal Cost:", params[["cost"]],
  "\nOptimal Kernel:", params[["kernel"]], "\n"
)
#   Optimal Cost: 6 
# Optimal Kernel: polynomial
```

## Kill AWS Resources

Now that we’ve completed our mini-analysis let’s make sure to kill all
our AWS resources. Since all we’ve done is launch EC2 instances, all
this consists of is making sure that the instances are all shut down.

``` r

ec2_client$stop_instances(
  InstanceIds = instance_ids(instance_req)
)
```
