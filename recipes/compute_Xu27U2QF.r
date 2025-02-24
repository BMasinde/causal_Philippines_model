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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
folder_path <- dkuManagedFolderPath("xcPrnvPS")
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)


# Construct the full file paths for the models
clas_file_path <- file.path(folder_path, "base_clas_min_model.rds")
wind_file_path  <- file.path(folder_path, "base_wind_model.rds")
rain_file_path  <- file.path(folder_path, "base_rain_model.rds")


# read the .rds model
base_clas_min_model  <- readRDS(clas_file_path)
base_wind_model  <- readRDS(wind_file_path)
base_rain_model  <- readRDS(rain_file_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predicting wind_max and rain & updating the base_test to df_base_test

df_base_test  <- base_test %>%
    mutate(
    wind_max_pred = predict(
      base_wind_model, newdata = base_test),
    rain_total_pred = predict(
      base_rain_model,
      newdata = base_test)
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# predict for damage_binary
# Make probability predictions for classification
y_preds_prob <- predict(base_clas_min_model, newdata = df_base_test, type = "prob")[,2]  # Probability of class 1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# AUC
# Compute AUC (better for classification)
auc_value <- auc(roc(df_base_test$damage_binary, y_preds_prob))
auc_value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# extracting probability that y_pred == 1
#y_preds_prob_1 <- y_preds_prob[ ,2]

## assigning final class based on threshold
y_pred <- ifelse(y_preds_prob > 0.5, 1, 0)

# using table function
conf_matrix <- table(predicted = y_pred,
                     actual = df_base_test$damage_binary
                     )
print(conf_matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

cat("test-set accuracy of minimal CLASSIFICATION SCM model:", accuracy, sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Extract recall and precision
# Compute confusion matrix
conf_matrix <- confusionMatrix(as.factor(y_pred), as.factor(df_base_test$damage_binary), positive = "1")
recall <- conf_matrix$byClass["Sensitivity"]  # Recall (Sensitivity)
precision <- conf_matrix$byClass["Precision"] # Precision

# metrics in a table
# Create a data frame with the metrics
metrics_df <- data.frame(
  Metric = c("Accuracy", "Recall", "Precision"),
  Value = c(accuracy, recall, precision)
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
#fwrite(metrics_df, file = file_path, row.names = FALSE)

dkuWriteDataset(metrics_df, "min_clas_metrics_df")

# Print message to confirm
print(paste("Metrics saved to:", metrics_folder_path))