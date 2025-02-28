# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Objective of recipe is to:
# Predict on the scm_min_clas_model on the test set
# Get the classification metrics on the test set

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
library(caret)
library(pROC) # For AUC calculation
library(dplyr)
library(data.table)
library(mlflow)
library(purrr)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predicting track_min_dist, wind_max and rain & updating the base_test to df_base_test

# df_base_test  <- base_test %>%
#     mutate(
#     track_min_dist_pred = predict(base_track_model, newdata = base_test),
#     wind_max_pred = predict(base_wind_model, newdata = base_test),
#     rain_total_pred = predict(base_rain_model, newdata = base_test),
#     wind_blue_ss = wind_max_pred * blue_ss_frac, # Updating interaction terms
#     wind_yellow_ss = wind_max_pred * yellow_ss_frac,
#     wind_orange_ss = wind_max_pred * orange_ss_frac,
#     wind_red_ss = wind_max_pred * red_ss_frac,
#     rain_blue_ss = rain_total_pred * blue_ls_frac,
#     rain_yellow_ss = rain_total_pred * yellow_ls_frac,
#     rain_orange_ss = rain_total_pred * orange_ls_frac,
#     rain_red_ss = rain_total_pred * red_ls_frac,

#     )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
folder_path <- dkuManagedFolderPath("xcPrnvPS")
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)

# Define the list of model names
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
models_list <- list()

# Loop over each model name to construct the file path and read the RDS file
for (model_name in model_names) {
  # Construct the file path for the model
  file_path <- file.path(folder_path, paste0("base_", model_name, "_model.rds"))

  # Read the model and store it in the list with the model name as the key
  models_list[[paste0("base_", model_name, "_model")]] <- readRDS(file_path)
}


# # Access the models using their names
# base_clas_full_model  <- models$base_clas_model
# base_wind_model  <- models$base_wind_model
# base_rain_model  <- models$base_rain_model
# base_track_model  <- models$base_track_model



# # Construct the full file paths for the models
# clas_file_path <- file.path(folder_path, "base_clas_full_model.rds")
# wind_file_path  <- file.path(folder_path, "base_wind_model.rds")
# rain_file_path  <- file.path(folder_path, "base_rain_model.rds")
# track_file_path  <- file.path(folder_path, "base_track_model.rds")


# # read the .rds model
# base_clas_full_model  <- readRDS(clas_file_path)
# base_wind_model  <- readRDS(wind_file_path)
# base_rain_model  <- readRDS(rain_file_path)
# base_track_model  <- readRDS(track_file_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply predictions efficiently

# Define models in a named list
col_models_list <- list(
  track_min_dist = models_list[["base_track_model"]],
  wind_max = models_list[["base_wind_model"]],
  rain_total = models_list[["base_rain_model"]],
  roof_strong_wall_strong = models_list[["base_roof_strong_wall_strong_model"]],
  roof_strong_wall_light = models_list[["base_roof_strong_wall_light_model"]],
  roof_strong_wall_salv = models_list[["base_roof_strong_wall_salv_model"]],
  roof_light_wall_strong = models_list[["base_roof_light_wall_strong_model"]],
  roof_light_wall_light = models_list[["base_roof_light_wall_light_model"]],
  roof_light_wall_salv = models_list[["base_roof_light_wall_salv_model"]],
  roof_salv_wall_strong = models_list[["base_roof_salv_wall_strong_model"]],
  roof_salv_wall_light = models_list[["base_roof_salv_wall_light_model"]],
  roof_salv_wall_salv = models_list[["base_roof_salv_wall_salv_model"]]
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_base_test <-  base_test %>%
  mutate(across(names(col_models_list), ~ predict(col_models_list[[cur_column()]],
                                             newdata = base_test), .names = "{.col}_pred"))

# Define wind and rain interaction variables
wind_fractions <- c("blue_ss_frac", "yellow_ss_frac", "orange_ss_frac", "red_ss_frac")
rain_fractions <- c("blue_ls_frac", "yellow_ls_frac", "orange_ls_frac", "red_ls_frac")

# Compute wind interaction terms dynamically
df_base_test <- df_base_test %>%
  mutate(across(all_of(wind_fractions), ~ . * wind_max_pred, .names = "wind_{.col}"),
         across(all_of(rain_fractions), ~ . * rain_total_pred, .names = "rain_{.col}"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_base_test$damage_binary_2 <- factor(df_base_test$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predict for damage_binary
# Make probability predictions for classification
y_preds_probs <- predict(models_list[["base_clas_full_model"]], newdata = df_base_test, type = "prob")[,2]  # Probability of class 1
#y_preds_probs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# AUC
# Compute AUC (better for classification)
auc_value <- auc(roc(df_base_test$damage_binary_2, y_preds_probs))
auc_value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# extracting probability that y_pred == 1
#y_preds_prob_1 <- y_preds_prob[ ,2]

## assigning final class based on threshold
y_pred <- ifelse(y_preds_probs > 0.5, 1, 0)

y_pred  <- factor(y_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels
# using table function
conf_matrix <- confusionMatrix(as.factor(y_pred),
                     df_base_test$damage_binary_2,
                     positive = "Damage_above_10"
                     )
print(conf_matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
models_list[["base_clas_full_model"]]$bestTune

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# logging in mflow:
# Logging the model and parameter using MLflow

# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "SCM - XGBOOST classification -CV (Test metircs)")

# Ensure that MLflow has only one run. Start MLflow run once.
run_name <- paste("XGBoost Run", Sys.time())  # Unique name using current time


# Start MLflow run
mlflow_start_run(nested = FALSE)

# Ensure the run ends even if an error occurs
#on.exit(mlflow_end_run(), add = TRUE)

# Extract the best parameters (remove AUC column)
#best_params_model <- best_params %>% # Remove AUC column if present
#    select(-AUC)

parameters_used  <- models_list[["base_clas_full_model"]]$bestTune

# Log each of the best parameters in MLflow
for (param in names(parameters_used)) {
  mlflow_log_param(param, parameters_used[[param]])
}

# Log the model type as a parameter
mlflow_log_param("model_type", "scm-xgboost-classification")

# predicting
y_preds_probs <- predict(models_list[["base_clas_full_model"]], newdata = df_base_test, type = "prob")[,2]  # Probability of class 1
y_pred <- ifelse(y_preds_probs > 0.5, 1, 0)

y_pred  <- factor(y_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# summarize results
conf_matrix <- confusionMatrix(as.factor(y_pred),
                     df_base_test$damage_binary_2,
                     positive = "Damage_above_10"
                     )

# accuracy
accuracy  <- conf_matrix$overall['Accuracy']

# Positive class = 1, precision, recall, and F1
# Extract precision, recall, and F1 score
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1_score <- conf_matrix$byClass['F1']
auc_value <- auc(roc(df_base_test$damage_binary_2, y_preds_probs))


# Log parameters and metrics
# mlflow_log_param("model_type", "scm-xgboost-classification")
mlflow_log_metric("accuracy", accuracy)
mlflow_log_metric("F1", f1_score)
mlflow_log_metric("Precision", precision)
mlflow_log_metric("Recall", recall)
#mlflow_log_metric("AUC", auc_value)


# Save model
#saveRDS(model, file = file.path(path_2_folder, "spam_clas_model.rds"))

# End MLflow run
mlflow_end_run()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Extract recall and precision
# Compute confusion matrix
conf_matrix <- confusionMatrix(as.factor(y_pred), df_base_test$damage_binary_2, positive = "Damage_above_10")
recall <- conf_matrix$byClass["Sensitivity"]  # Recall (Sensitivity)
precision <- conf_matrix$byClass["Precision"] # Precision
f1_score  <- conf_matrix$byClass["F1"]
accuracy  <- conf_matrix$overall['Accuracy']

# metrics in a table
# Create a data frame with the metrics
metrics_df <- data.frame(
  Metric = c("Accuracy", "Recall", "Precision", "F1", "AUC"),
  Value = c(accuracy, recall, precision, f1_score, auc_value)
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
metrics_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
metrics_folder_path <- dkuManagedFolderPath("Xu27U2QF")

# Saving the predicted values
# Define file path
file_path <- file.path(metrics_folder_path, "model_metrics.csv")

# Write to CSV
fwrite(metrics_df, file = file_path, row.names = FALSE)

#dkuWriteDataset(metrics_df, "min_clas_metrics_df")

# Print message to confirm
print(paste("Metrics saved to:", metrics_folder_path))