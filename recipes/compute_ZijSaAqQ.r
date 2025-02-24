# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Libraries
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
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)

base_wind_model <- rpart(wind_max ~ track_min_dist,
                       data = base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for rain speed
# rain_total = f(track_min_dist, eps)

base_rain_model <- rpart(rain_total ~ track_min_dist,
                       data = base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adding the predicted parents' to the training dataset

df_base_train <- base_train %>%
  mutate(wind_max_pred = predict(base_wind_model,
                                 newdata = base_train),
         rain_total_pred = predict(base_rain_model,
                                   newdata = base_train)
         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# parameter tuning
# Define a grid of hyperparameters
cp_values <- seq(0.0001, 0.05, by = 0.005)
maxdepth_values <- c(3, 5, 7, 10)
minsplit_values <- c(10, 20, 30, 40)
minbucket_values <- c(5, 10, 20)

# Create an empty list to store results
# Create an empty list to store results
results <- data.frame(cp = numeric(), maxdepth = numeric(),
                      minsplit = numeric(), minbucket = numeric(), RMSE = numeric())

# predicting for wind and rainfall for the validation dataset
df_val_base_tune <- base_validation %>%
  mutate(
    wind_max_pred = predict(
      base_wind_model, newdata = base_validation),
    rain_total_pred = predict(
      base_rain_model,
      newdata = base_validation)
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
          damage_perc ~ wind_max_pred +
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
          method = "anova",  # Regression
          control = rpart.control(cp = cp, maxdepth = maxdepth,
                                  minsplit = minsplit, minbucket = minbucket)
        )

        # Make predictions on the validation set
        val_predictions <- predict(model, newdata = df_val_base_tune)

        # Compute RMSE
        rmse_value <- rmse(df_val_base_tune$damage_perc, val_predictions)

        # Store results
        results <- rbind(results, data.frame(cp, maxdepth, minsplit, minbucket, RMSE = rmse_value))
      }
    }
  }
}

# Print the best hyperparameter combination
best_params <- results[which.min(results$RMSE), ]
print(best_params)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training based on tuned parameters

# Combine Training and Validation datasets for final training

final_training_df  <- rbind(df_base_train,
                           df_val_base_tune)


damage_fit_reg_min <- rpart(damage_perc ~ wind_max_pred +
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
                              method = "anova",
                              control = rpart.control(cp = best_params$cp,
                                                      maxdepth = best_params$maxdepth,
                                                      minsplit = best_params$minsplit,
                                                      minbucket = best_params$minbucket),
                              data = final_training_df
                         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# RMSE on the trainset (training + validation)
# Compute RMSE

damage_pred  <- predict(damage_fit_reg_min, newdata = final_training_df)
rmse_value <- rmse(final_training_df$damage_perc, damage_pred)
rmse_value

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
managed_folder_path <- dkuManagedFolderPath("ZijSaAqQ")

saveRDS(damage_fit_reg_min, file = paste0(managed_folder_path, "/base_reg_min_model.rds"))

saveRDS(base_wind_model, file = paste0(managed_folder_path, "/base_wind_model.rds"))

saveRDS(base_rain_model, file = paste0(managed_folder_path, "/base_rain_model.rds"))