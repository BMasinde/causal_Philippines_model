# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Libraries
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(pROC) # For AUC calculation
library(data.table)
library(mlflow)
library(reticulate)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
py_config()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
df_base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)

base_wind_model <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for rain speed
# rain_total = f(track_min_dist, eps)

base_rain_model <- rpart(rain_total ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adding the predicted parents' to the training dataset

df_base_train <- df_base_train %>%
  mutate(wind_max_pred = predict(base_wind_model,
                                 newdata = df_base_train),
         rain_total_pred = predict(base_rain_model,
                                   newdata = df_base_train)
         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# parameter tuning
# Define a grid of hyperparameters
cp_values <- seq(0.0001, 0.05, by = 0.005)
maxdepth_values <- c(3, 5, 7, 10)
minsplit_values <- c(10, 20, 30, 40)
minbucket_values <- c(5, 10, 20)

# Create an empty list to store results
results_list <- list()

# predicting for wind and rainfall for the validation dataset
df_val_base_tune <- df_base_validation %>%
  mutate(
    wind_max_pred = predict(
      base_wind_model, newdata = df_base_validation),
    rain_total_pred = predict(
      base_rain_model,
      newdata = df_base_validation)
    )

# Train the model using manual grid search
grid_id <- 1  # Index for list storage

# Iterate over all combinations of hyperparameters
for (cp in cp_values) {
  for (maxdepth in maxdepth_values) {
    for (minsplit in minsplit_values) {
      for (minbucket in minbucket_values) {

        # Train the model with specific hyperparameters
        model <- rpart(
          damage_binary ~ wind_max_pred +
            rain_total_pred +
            roof_strong_wall_strong +
            roof_strong_wall_light +
            roof_strong_wall_salv +
            roof_light_wall_strong +
            roof_light_wall_light +
            roof_light_wall_salv +
            roof_salv_wall_strong +
            roof_salv_wall_light +
            roof_salv_wall_salv +
            ls_risk_pct +
            ss_risk_pct +
            wind_blue_ss +
            wind_yellow_ss +
            wind_orange_ss +
            wind_red_ss +
            rain_blue_ss +
            rain_yellow_ss +
            rain_orange_ss +
            rain_red_ss,
          data = df_base_train,
          method = "class",  # classification
          control = rpart.control(cp = cp, maxdepth = maxdepth,
                                  minsplit = minsplit, minbucket = minbucket)
        )

        # Make probability predictions for classification
        val_predictions <- predict(model, newdata = df_val_base_tune, type = "prob")[,2]  # Probability of class 1

        # Compute AUC (better for classification)
        auc_value <- auc(df_val_base_tune$damage_binary, val_predictions)

        # Store results efficiently in a list
        results_list[[grid_id]] <- data.frame(cp, maxdepth, minsplit, minbucket, AUC = auc_value)
        grid_id <- grid_id + 1
      }
    }
  }
}

# Convert list to data frame
results <- rbindlist(results_list)

# Print the best hyperparameter combination (highest AUC)
best_params <- results[which.max(results$AUC), ]
print(best_params)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training based on tuned parameters

# Combine Training and Validation datasets for final training

final_training_df  <- rbind(df_base_train,
                           df_val_base_tune)


damage_fit_class_min <- rpart(damage_binary ~ wind_max_pred +
                              rain_total_pred +
                              roof_strong_wall_strong +
                              roof_strong_wall_light +
                              roof_strong_wall_salv +
                              roof_light_wall_strong +
                              roof_light_wall_light +
                              roof_light_wall_salv +
                              roof_salv_wall_strong +
                              roof_salv_wall_light +
                              roof_salv_wall_salv +
                              ls_risk_pct +
                              ss_risk_pct +
                              wind_blue_ss +
                              wind_yellow_ss +
                              wind_orange_ss +
                              wind_red_ss +
                              rain_blue_ss +
                              rain_yellow_ss +
                              rain_orange_ss +
                              rain_red_ss,
                              method = "class",
                              control = rpart.control(cp = best_params$cp,
                                                      maxdepth = best_params$maxdepth,
                                                      minsplit = best_params$minsplit,
                                                      minbucket = best_params$minbucket),
                              data = final_training_df
                         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# testing on the training datasets (training + validation)

## Outcome prediction on the final_training_df dataset
## default function predict returns class probabilities (has two columns)
y_pred_probs <- predict(damage_fit_class_min,
                  newdata = final_training_df)

## extracting probability that y_pred == 1
y_pred_prob_1 <- y_pred_probs[ ,2]

## assigning final class based on threshold
y_pred <- ifelse(y_pred_prob_1 > 0.5, 1, 0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# using table function
conf_matrix <- table(predicted = y_pred,
                     actual = final_training_df$damage_binary
                     )
print(conf_matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

cat("test-set accuracy of minimal SCM model:", accuracy, sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#' Loggint the model and parameter using MLflow
# Start MLflow Run

# Configure reticulate to use the Python environment with MLflow
# use_python(Sys.which("python3"))

# mlflow <- import("mlflow")

# Assuming 'damage_fit_class_min' is your R model object
# Load your R model (saved as .rds file)
# model <- readRDS("path/to/your/model.rds")

# Assuming you have some hyperparameters for the model (example)
#hyperparameters <- list(
#  cp = best_params$cp,
#  maxdepth = best_params$maxdepth,
#  minsplit = best_params$minsplit,
#  minbucket = best_params$minbucket
#)

# Assuming 'accuracy' is the accuracy score of your model (example)
#accuracy <- 0.85  # Replace with your actual accuracy score

# Function to log the R model with hyperparameters and accuracy
#log_model_to_mlflow <- function(model, accuracy, hyperparameters) {
    # Start an MLflow run
#  mlflow$start_run()

  # Log hyperparameters
#  mlflow$log_param("cp", hyperparameters$cp)
#  mlflow$log_param("maxdepth", hyperparameters$maxdepth)
#  mlflow$log_param("minsplit", hyperparameters$minsplit)
#  mlflow$log_param("minbucket", hyperparameters$minbucket)


  # Log model accuracy
#  mlflow$log_metric("accuracy", accuracy)

  # Save the model to the managed folder path in Dataiku DSS
#  managed_folder_path <- dkuManagedFolderPath("xcPrnvPS")
#  model_path <- paste0(managed_folder_path, "/base_clas_min_model.rds")

  # Save the model as an RDS file in the managed folder
#  saveRDS(model, file = model_path)

  # Log the saved model as an artifact in MLflow
#  mlflow$log_artifact(model_path)

    # End the MLflow run
#  mlflow$end_run()
#}

# Log the model, accuracy, and hyperparameters to MLflow
#log_model_to_mlflow(damage_fit_class_min, accuracy, hyperparameters)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
managed_folder_path <- dkuManagedFolderPath("xcPrnvPS")
saveRDS(damage_fit_class_min, file = paste0(managed_folder_path, "/base_clas_min_model.rds"))
saveRDS(base_wind_model, file = paste0(managed_folder_path, "/base_wind_model.rds"))
saveRDS(base_rain_model, file = paste0(managed_folder_path, "/base_wind_model.rds"))