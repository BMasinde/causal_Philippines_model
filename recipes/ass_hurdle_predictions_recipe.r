# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(data.table)
library(mlflow)
library(reticulate)
library(Metrics)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
ass_XGBOOST_base_reg <- dkuManagedFolderPath("DFdq9elg")
ass_XGBOOST_classifier <- dkuManagedFolderPath("fZ8zhmA4")
ass_XGBOOST_trunc_reg <- dkuManagedFolderPath("ZGOs5kwX")


# Construct the file path for the model
base_reg_file_path <- file.path(ass_XGBOOST_base_reg, "damage_fit_reg_base.rds")
trunc_reg_file_path <- file.path(ass_XGBOOST_trunc_reg, "trunc_damage_fit_reg.rds")
clas_file_path <- file.path(ass_XGBOOST_classifier, "ass_XGBOOST_class.rds")

# Read the .rds models
base_reg <- readRDS(base_reg_file_path)
trunc_reg <- readRDS(trunc_reg_file_path)
clas_model  <- readRDS(clas_file_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# reading the test datasets.
# Reading base_test data
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)

# Redaing trunc_test data
truncated_test <- dkuReadDataset("truncated_test", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# HURDLE METHOD FUNCTION
#' Title: Predict the building damage % from TCs
#'
#' Function takes the test data & trained models and returns predicted building damages.
#'
#' @param df A dataframe for prediction (can be the test set for testing hurdle method)
#' @param ass_clas_model The trained model for classification
#' @param ass_base_model A list of the SCM models for the base regression
#' @param ass_trunc_model A list of SCM models for the high-impact regression
#'
#'

assPredictDamage <- function(df, ass_clas_model, ass_base_model, ass_trunc_model, threshold) {

  # factors cleaning for classification task
  df$damage_binary_2 <- factor(df$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

  ## Step 1: Predict the class label (whether the damage will exceed the threshold)
  ## class_model should return predicted classes and not probs.
  ## type = "prob" for custom threshold specification
  prob_pred <- predict(ass_clas_model, df, type = "prob")[,2]  # Probability of class 1
  ## assigning final class based on threshold
  class_pred <- ifelse(prob_pred > threshold, 1, 0) # low threhold of 0.35 can be changed to 0.65/0.75

  class_pred  <- factor(class_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

  ## Step 2: Predict the base damage percentage using the base regression model (for low impact cases)
  ## should return the predicted damage percentages
  base_pred <- predict(ass_base_model, df)

  ## Step 3: Predict the high-impact damage percentage using the high-impact

  high_pred <- predict(ass_trunc_model, df)

  # Step 4: Apply the hurdle method logic
  predicted_damage <- ifelse(class_pred == "Damage_above_10", high_pred, base_pred)

  # Return the predicted damage
  return(predicted_damage)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# predicting on base test set data
## because we already implemented the hurdle method
df_test <- bind_rows(
  base_test,
  truncated_test
)

# setting threshold for classification step
threshold = 0.35

preds <- assPredictDamage(df = df_test, ass_clas_model = clas_model,
  ass_base_model = base_reg, ass_trunc_model = trunc_reg ,threshold = threshold)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define bin edges
# Define bin edges
bins <- c(0, 0.00009, 1, 10, 50, 100)

# Assign data to bins
bin_labels <- cut(df_test$damage_perc, breaks = bins, include.lowest = TRUE, right = TRUE)

# Create a data frame with actual, predicted, and bin labels
data <- data.frame(
  actual = df_test$damage_perc,
  predicted = preds,
  bin = bin_labels
)

# Calculate RMSE per bin
unique_bins <- levels(data$bin) # Get unique bin labels
rmse_by_bin <- data.frame(bin = unique_bins, rmse = NA, count = NA) # Initialize results data frame

for (i in seq_along(unique_bins)) {
  bin_data <- data[data$bin == unique_bins[i], ] # Filter data for the current bin
  rmse_by_bin$rmse[i] <- sqrt(mean((bin_data$actual - bin_data$predicted)^2, na.rm = TRUE)) # Calculate RMSE
  rmse_by_bin$count[i] <- nrow(bin_data) # Count observations in the bin
}

# Calculate weighted average RMSE
total_count <- sum(rmse_by_bin$count, na.rm = TRUE)
w_avg  <- sum(rmse_by_bin$rmse * rmse_by_bin$count)/total_count

# Display RMSE by bin
print(rmse_by_bin)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
w_avg

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Log metrics using MLFLOW
# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "U-SCM - Hurlde - CV (Test metircs)")

# Ensure that MLflow has only one run. Start MLflow run once.
run_name <- paste("Hurdle Run", Sys.time())  # Unique name using current time

as.data.frame(rmse_by_bin)
RMSE_09 <- rmse_by_bin[1, "rmse"]
RMSE_1 <- rmse_by_bin[2, "rmse"]
RMSE_10 <-  rmse_by_bin[3, "rmse"]
RMSE_50 <- rmse_by_bin[4, "rmse"]
RMSE_100 <- rmse_by_bin[5, "rmse"]

# Log threshold & binned RMSE metrics
mlflow_log_metric("thresh", threshold)
mlflow_log_metric("RMSE_09", RMSE_09)
mlflow_log_metric("RMSE_1", RMSE_1)
mlflow_log_metric("RMSE_10", RMSE_10)
mlflow_log_metric("RMSE_50", RMSE_50)
mlflow_log_metric("RMSE_100", RMSE_100)
mlflow_log_metric("w_avg", w_avg)
# End MLflow run
mlflow_end_run()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
ass_hurdle_predictions <- dkuManagedFolderPath("K9arRZ6K")