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
# Final Testing data ---------------------------------------------------------
# Reading base_test data
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)

# Redaing trunc_test data
truncated_test <- dkuReadDataset("truncated_test", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
clas_folder_path <- dkuManagedFolderPath("xcPrnvPS")
base_reg_path <- dkuManagedFolderPath("ZijSaAqQ")
trunc_reg_path <- dkuManagedFolderPath("dL4i4SKb")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the list of model names for the base model
model_names <- c("clas_full",
                 "track",
                 "wind",
                 "rain",
                 "track",
                 "roof_strong_wall_strong",
                 "roof_strong_wall_light",
                 "roof_strong_wall_salv",
                 "roof_light_wall_strong",
                 "roof_light_wall_light",
                 "roof_light_wall_salv",
                 "roof_salv_wall_strong",
                 "roof_salv_wall_light",
                 "roof_salv_wall_salv"
                )

# Create a named list to store the models
base_models_list <- list()

# Loop over each model name to construct the file path and read the RDS file
for (model_name in model_names) {
  # Construct the file path for the model
  file_path <- file.path(clas_folder_path, paste0("base_", model_name, "_model.rds"))

  # Read the model and store it in the list with the model name as the key
  base_models_list[[paste0("base_", model_name, "_model")]] <- readRDS(file_path)
}


base_models_list$base_reg_model  <- readRDS(
     file.path(base_reg_path, "base_reg_model.rds")
 )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# reading truncated models
trunc_model_names <- c("reg",
                 "track",
                 "wind",
                 "rain",
                 "roof_strong_wall_strong",
                 "roof_strong_wall_light",
                 "roof_strong_wall_salv",
                 "roof_light_wall_strong",
                 "roof_light_wall_light",
                 "roof_light_wall_salv",
                 "roof_salv_wall_strong",
                 "roof_salv_wall_light",
                 "roof_salv_wall_salv"
                )

# Create a named list to store the models
trunc_models_list <- list()

# Loop over each model name to construct the file path and read the RDS file
for (model_name in trunc_model_names) {
  # Construct the file path for the model
  file_path <- file.path(trunc_reg_path, paste0("trunc_", model_name, "_model.rds"))

  # Read the model and store it in the list with the model name as the key
  trunc_models_list[[paste0("trunc_", model_name, "_model")]] <- readRDS(file_path)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
names(trunc_models_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the base .rds models --------------------------------------

# reading classification model
# base_clas_min_model  <- readRDS(
#     file.path(clas_folder_path, "base_clas_min_model.rds")
# )

# # reading the base wind and rain models
# base_track_model  <- readRDS(
#     file.path(base_reg_path, "base_track_model.rds")
# )

# base_wind_model  <- readRDS(
#     file.path(base_reg_path, "base_wind_model.rds")
# )
# base_rain_model  <- readRDS(
#     file.path(base_reg_path, "base_rain_model.rds")
# )

# # base regression model
# base_reg_model  <- readRDS(
#     file.path(base_reg_path, "base_reg_min_model.rds")
# )

# # Reading the truncated .rds models -------------------------------

# # reading the truncated track, wind and rain models
# trunc_track_model  <- readRDS(
#     file.path(trunc_reg_path, "trunc_track_model.rds")
# )

# trunc_wind_model  <- readRDS(
#     file.path(trunc_reg_path, "trunc_wind_model.rds")
# )
# trunc_rain_model  <- readRDS(
#     file.path(trunc_reg_path, "trunc_rain_model.rds")
# )

# # base regression model
# trunc_reg_model  <- readRDS(
#     file.path(trunc_reg_path, "trunc_reg_min_model.rds")
# )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Putting models into lists

# ## trained SCM models on base data
# base_models <- list(
#   "track_model" = base_track_model
#   "wind_model" = base_wind_model,
#   "rain_model" = base_rain_model,
#   "base_reg_model" = base_reg_model
# )

# ## trained SCM models on high impact data (damage >= 10)

# high_models <- list(
#   "track_model_high" = trunc_track_model,
#   "wind_model_high" = trunc_wind_model,
#   "rain_model_high" = trunc_rain_model,
#   "high_reg_model" =  trunc_reg_model
# )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# HURDLE METHOD FUNCTION
#' Title: Predict the building damage % from TCs
#'
#' Function takes the test data & trained models and returns predicted building damages.
#'
#' @param df A dataframe for prediction (can be the test set for testing hurdle method)
#' @param class_model The trained model for classification
#' @param scm_models_base A list of the SCM models for the base regression
#' @param scm_models_high A list of SCM models for the high-impact regression
#'
#'

predictDamage <- function(df, scm_models_base, scm_models_high, threshold) {
    
base_col_models_list <- list(
  track_min_dist = scm_models_base[["base_track_model"]],
  wind_max = scm_models_base[["base_wind_model"]],
  rain_total = scm_models_base[["base_rain_model"]],
  roof_strong_wall_strong = scm_models_base[["base_roof_strong_wall_strong_model"]],
  roof_strong_wall_light = scm_models_base[["base_roof_strong_wall_light_model"]],
  roof_strong_wall_salv = scm_models_base[["base_roof_strong_wall_salv_model"]],
  roof_light_wall_strong = scm_models_base[["base_roof_light_wall_strong_model"]],
  roof_light_wall_light = scm_models_base[["base_roof_light_wall_light_model"]],
  roof_light_wall_salv = scm_models_base[["base_roof_light_wall_salv_model"]],
  roof_salv_wall_strong = scm_models_base[["base_roof_salv_wall_strong_model"]],
  roof_salv_wall_light = scm_models_base[["base_roof_salv_wall_light_model"]],
  roof_salv_wall_salv = scm_models_base[["base_roof_salv_wall_salv_model"]]
) 

  ## common predictions btw class & base regression
  df <-  df %>%
  mutate(across(names(base_col_models_list), ~ predict(base_col_models_list[[cur_column()]],
                                             newdata = df), .names = "{.col}_pred"))
  # factors cleaning for classification task
  df$damage_binary_2 <- factor(df$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

  ## Step 1: Predict the class label (whether the damage will exceed the threshold)
  ## class_model should return predicted classes and not probs.
  ## class_model expects variables "wind_max_pred" and "rain_total_pred" in dataframe df
  ## type = "prob" for custom threshold specification
  prob_pred <- predict(scm_models_base$base_clas_full_model, df, type = "prob")[,2]  # Probability of class 1
  ## assigning final class based on threshold
  class_pred <- ifelse(prob_pred > threshold, 1, 0) # low threhold of 0.35 can be changed to 0.65/0.75

  class_pred  <- factor(class_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

  ## Step 2: Predict the base damage percentage using the base regression model (for low impact cases)
  ## base expects variables "wind_max_pred" and "rain_total_pred" in dataframe df
  ## should return the predicted damage percentages
  base_pred <- predict(scm_models_base$base_reg_model, df)

  ## Step 3: Predict the high-impact damage percentage using the high-impact
  ### SCM models (for high impact cases)
  ## wind and rainfall predictions are based on high impact data (damage >= 10)
    
  trunc_col_models_list <- list(
  track_min_dist = scm_models_high[["trunc_track_model"]],
  wind_max = scm_models_high[["trunc_wind_model"]],
  rain_total = scm_models_high[["trunc_rain_model"]],
  roof_strong_wall_strong = scm_models_high[["trunc_roof_strong_wall_strong_model"]],
  roof_strong_wall_light = scm_models_high[["trunc_roof_strong_wall_light_model"]],
  roof_strong_wall_salv = scm_models_high[["trunc_roof_strong_wall_salv_model"]],
  roof_light_wall_strong = scm_models_high[["trunc_roof_light_wall_strong_model"]],
  roof_light_wall_light = scm_models_high[["trunc_roof_light_wall_light_model"]],
  roof_light_wall_salv = scm_models_high[["trunc_roof_light_wall_salv_model"]],
  roof_salv_wall_strong = scm_models_high[["trunc_roof_salv_wall_strong_model"]],
  roof_salv_wall_light = scm_models_high[["trunc_roof_salv_wall_light_model"]],
  roof_salv_wall_salv = scm_models_high[["trunc_roof_salv_wall_salv_model"]]
) 
  # add the predictions of wind and rainfall to the dataframe df
  df2 <- df %>%
      mutate(across(names(trunc_col_models_list), ~ predict(trunc_col_models_list[[cur_column()]],
                                             newdata = df), .names = "{.col}_pred"))

  high_pred <- predict(scm_models_high$trunc_reg_model, df2)

  # Step 4: Apply the hurdle method logic
  predicted_damage <- ifelse(class_pred == "Damage_above_10", high_pred, base_pred)

  # Return the predicted damage
  return(predicted_damage)
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
names(base_models_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predicting on base test set data
## because we already implemented the hurdle method
df_test <- bind_rows(
  base_test,
  truncated_test
)

# setting threshold for classification step
threshold = 0.35

preds <- predictDamage(df = df_test, scm_models_base = base_models_list,
  scm_models_high = trunc_models_list, threshold = threshold
  
)

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

# Display RMSE by bin
print(rmse_by_bin)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Log metrics using MLFLOW
# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "SCM - Hurlde - CV (Test metircs)")

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
# End MLflow run
mlflow_end_run()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Writing binned RMSE to folder

# Recipe outputs
folder_path <- dkuManagedFolderPath("5NPBmWH1")

# Saving the predicted values
# Define file path

# Generate timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Define file path with timestamp
file_path <- file.path(folder_path, paste0("rmse_by_bin_", timestamp, ".csv"))

# Write to CSV
fwrite(as.data.frame(rmse_by_bin), file = file_path, row.names = FALSE)