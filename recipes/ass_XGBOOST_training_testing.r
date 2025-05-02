# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(pROC) # For AUC calculation
library(data.table)
library(mlflow)
library(reticulate)
library(Matrix)
library(purrr) # useful for code optimization
library(themis)
library(doMC)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Combining train and validation datasets to one
# Because we are going to use CV to train the models later
# naming it df_base_train to remain consistent with df naming
df_base_train  <- rbind(base_train, base_validation)

cat("number of rows in combined train data:", nrow(df_base_train), sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Ensure target variable is a factor
# Ensure the target variable is a factor with valid names
df_base_train$damage_binary_2 <- factor(df_base_train$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# setting seed for reproducibility
set.seed(1234)

tune_grid <- expand.grid(
  nrounds = c(47,50, 60,70), # early stopping does not work, we still need to specify nrounds
  max_depth = c(2, 3, 4, 6),
  eta = c(0.09, 0.1, 0.11, 0.12),
  gamma = c(0, 1, 2, 3, 4),
  colsample_bytree = c(0.9, 1.0, 1.1),
  min_child_weight = c(2, 3, 4),
  subsample = c(0.5, 0.6, 0.7, 0.8)
)


# Set up train control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 7,
  classProbs = TRUE,  # Needed for AUC calculation
  summaryFunction = twoClassSummary,
  sampling = "smote", # caret automatically identifies minority class
  search = "random" # random selection of the expanded grid
)

# Detect and register the number of available cores (use all but one)
num_cores <- parallel::detectCores() - 2
registerDoMC(cores = num_cores)  # Enable parallel processing


# Measure the time for a code block to run
system.time({
    # Train the model using grid search with 10-fold CV

    xgb_model <- train(damage_binary_2 ~ track_min_dist + 
                       wind_max +
                       rain_total +
                       roof_strong_wall_strong +
                       roof_strong_wall_light +
                       roof_strong_wall_salv +
                       roof_light_wall_strong +
                       roof_light_wall_light +
                       roof_light_wall_salv +
                       roof_salv_wall_strong +
                       roof_salv_wall_light +
                       roof_salv_wall_salv +
                       blue_ss_frac +
                       yellow_ss_frac +
                       orange_ss_frac +
                       red_ss_frac +
                       blue_ls_frac +
                       yellow_ls_frac +
                       orange_ls_frac +
                       red_ls_frac,
        data = df_base_train,
        method = "xgbTree",
        trControl = train_control,
        tuneGrid = tune_grid,
        metric = "ROC" # "xgbTree" does not support other metrics for classification tasks (e.g., Kappa or F1)
    )
    Sys.sleep(2)  # This is just an example to simulate a delay
})

# Print best parameters
print(xgb_model$bestTune)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Extract the best parameters (remove AUC column)
best_params_model <- xgb_model$bestTune

damage_fit_class_full <- train(damage_binary_2 ~ track_min_dist + 
                               wind_max +
                               rain_total +
                               roof_strong_wall_strong +
                               roof_strong_wall_light +
                               roof_strong_wall_salv +
                               roof_light_wall_strong +
                               roof_light_wall_light +
                               roof_light_wall_salv +
                               roof_salv_wall_strong +
                               roof_salv_wall_light +
                               roof_salv_wall_salv +
                               blue_ss_frac +
                               yellow_ss_frac +
                               orange_ss_frac +
                               red_ss_frac +
                               blue_ls_frac +
                               yellow_ls_frac +
                               orange_ls_frac +
                               red_ls_frac, 
          data = df_base_train, # USE TRAINING AND VALIDATION SETS COMBINED
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = best_params_model # USE BEST PARAMETER
        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames(df_base_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# testing on the training datasets (training + validation)

## Outcome prediction on the final_training_df dataset
## default function predict returns class probabilities (has two columns)
y_pred <- predict(damage_fit_class_full,
                  newdata = df_base_train)


levels(y_pred)


# using table function
conf_matrix <- confusionMatrix(y_pred,
                     df_base_train$damage_binary_2, # remember to use damage_binary_2
                     positive = "Damage_above_10"
                     )
conf_matrix


accuracy <- conf_matrix$overall['Accuracy']

cat("test-set accuracy of associational XGBOOST:", accuracy, sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Logging the model and parameter using MLflow

# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "ass-XGBOOST classification - CV (Training metircs)")

# Ensure that MLflow has only one run. Start MLflow run once.
run_name <- paste("XGBoost Run", Sys.time())  # Unique name using current time


# Start MLflow run
mlflow_start_run(nested = FALSE)

# Ensure the run ends even if an error occurs
#on.exit(mlflow_end_run(), add = TRUE)

# Extract the best parameters (remove AUC column)
best_params_model <- xgb_model$bestTune

# Log each of the best parameters in MLflow
for (param in names(best_params_model)) {
  mlflow_log_param(param, best_params_model[[param]])
}

# Log the model type as a parameter
mlflow_log_param("model_type", "ass-xgboost-classification")

damage_fit_class_full <- train(damage_binary_2 ~ track_min_dist + 
                               wind_max +
                               rain_total +
                               roof_strong_wall_strong +
                               roof_strong_wall_light +
                               roof_strong_wall_salv +
                               roof_light_wall_strong +
                               roof_light_wall_light +
                               roof_light_wall_salv +
                               roof_salv_wall_strong +
                               roof_salv_wall_light +
                               roof_salv_wall_salv +
                               blue_ss_frac +
                               yellow_ss_frac +
                               orange_ss_frac +
                               red_ss_frac +
                               blue_ls_frac +
                               yellow_ls_frac +
                               orange_ls_frac +
                               red_ls_frac, 
          data = df_base_train, # USE TRAINING AND VALIDATION SETS COMBINED
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = best_params_model # USE BEST PARAMETER
        )


# summarize results
conf_matrix <- confusionMatrix(y_pred,
                     df_base_train$damage_binary_2,
                     positive = "Damage_above_10"
                     )

# accuracy
accuracy  <- conf_matrix$overall['Accuracy']

# Positive class = 1, precision, recall, and F1
# Extract precision, recall, and F1 score
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1_score <- conf_matrix$byClass['F1']


# Log parameters and metrics
# mlflow_log_param("model_type", "scm-xgboost-classification")
mlflow_log_metric("accuracy", accuracy)
mlflow_log_metric("F1", f1_score)
mlflow_log_metric("Precision", precision)
mlflow_log_metric("Recall", recall)


# Save model
#saveRDS(model, file = file.path(path_2_folder, "spam_clas_model.rds"))

# End MLflow run
mlflow_end_run()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Testing on test dataset ----------------------------------------------------------------------------------------
base_test$damage_binary_2 <- factor(base_test$damage_binary,
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# predict for damage_binary
# Make probability predictions for classification
y_preds_probs <- predict(damage_fit_class_full, 
                         newdata = base_test, type = "prob")[,2]  # Probability of class 1

# AUC
# Compute AUC (better for classification)
auc_value <- auc(roc(base_test$damage_binary_2, y_preds_probs))
auc_value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## assigning final class based on threshold
y_pred <- ifelse(y_preds_probs > 0.35, 1, 0)

y_pred  <- factor(y_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels
# using table function
conf_matrix <- confusionMatrix(as.factor(y_pred),
                     base_test$damage_binary_2,
                     positive = "Damage_above_10"
                     )
print(conf_matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "ass- XGBOOST classification - CV (Test metircs)")

# Ensure that MLflow has only one run. Start MLflow run once.
run_name <- paste("XGBoost Run", Sys.time())  # Unique name using current time


# Start MLflow run
mlflow_start_run(nested = FALSE)

# Ensure the run ends even if an error occurs
#on.exit(mlflow_end_run(), add = TRUE)

# Extract the best parameters (remove AUC column)
#best_params_model <- best_params %>% # Remove AUC column if present
#    select(-AUC)

parameters_used  <- xgb_model$bestTune

# Log each of the best parameters in MLflow
for (param in names(parameters_used)) {
  mlflow_log_param(param, parameters_used[[param]])
}

# Log the model type as a parameter
mlflow_log_param("model_type", "ass-xgboost-classification")

# predicting
y_preds_probs <- predict(damage_fit_class_full, newdata = base_test, type = "prob")[,2]  # Probability of class 1
y_pred <- ifelse(y_preds_probs > 0.35, 1, 0)

y_pred  <- factor(y_pred, levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# summarize results
conf_matrix <- confusionMatrix(as.factor(y_pred),
                     base_test$damage_binary_2,
                     positive = "Damage_above_10"
                     )

# accuracy
accuracy  <- conf_matrix$overall['Accuracy']

# Positive class = 1, precision, recall, and F1
# Extract precision, recall, and F1 score
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1_score <- conf_matrix$byClass['F1']
auc_value <- auc(roc(base_test$damage_binary_2, y_preds_probs))


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
# test set confusion matrix
conf_matrix

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get the false positves by regions (Luzon, Visayas, Mindanao)
# confusion matrix by regions
# Make sure the grouping variable is a factor
# Make sure island_groups is a factor
base_test$island_groups <- as.factor(base_test$island_groups)

# Loop through each group and generate a confusion matrix
for (grp in levels(base_test$island_groups)) {
  
  # Subset data for the current group
  group_indices <- base_test$island_groups == grp
  y_true_group <- base_test$damage_binary_2[group_indices]
  y_pred_group <- y_pred[group_indices]
  
  # Generate and print confusion matrix
  cat("Confusion Matrix for Island Group:", grp, "\n")
  print(confusionMatrix(y_pred_group, y_true_group, positive = "Damage_above_10"))
  cat("\n")
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
ass_XGBOOST_classifier <- dkuManagedFolderPath("fZ8zhmA4")

# save the trained truncated model
model_filename <- "ass_XGBOOST_class.rds"

saveRDS(damage_fit_class_full, file = file.path(ass_XGBOOST_classifier, model_filename))