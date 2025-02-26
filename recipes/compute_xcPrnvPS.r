# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Turning warning messages off
options(warn = 0) # i don't care for the messages

# Libraries
library(dataiku)
library(rpart)
library(dplyr)
library(caret)
library(pROC) # For AUC calculation
library(data.table)
library(mlflow)
library(reticulate)
library(Matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
df_base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training track_min_dist ~ island_groups
base_track_model  <- rpart(track_min_dist  ~ island_groups,
                          data = df_base_train,
                          method = "anova")
# new values for trac_min_dist
df_base_train  <- df_base_train %>%
    mutate(track_min_dist = predict(base_track_model,
                                   newdata = df_base_train)
          )

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
  mutate(track_min_dist = predict(base_track_model, newdata = df_base_train)) # predicted min_dist
  mutate(wind_max_pred = predict(base_wind_model,
                                 newdata = df_base_train),
         rain_total_pred = predict(base_rain_model,
                                   newdata = df_base_train)
         )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Parameter tuning

# Define tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 100, 150),       # Number of boosting rounds
  max_depth = c(3, 6, 9),          # Maximum tree depth
  eta = c(0.01, 0.1, 0.3),         # Learning rate
  gamma = 0,                       # Minimum loss reduction
  colsample_bytree = 0.8,          # Feature selection rate
  min_child_weight = 1,            # Minimum instance weight
  subsample = 0.8                  # Sample ratio per boosting round
)


# Create an empty list to store results
results_list <- list()

# Extra data prep
# Ensure target variable is a factor for classification
df_base_train$damage_binary <- as.factor(df_base_train$damage_binary)

# predicting for wind and rainfall for the validation dataset
df_val_base_tune <- df_base_validation %>%
  mutate(track_min_dist = predict(base_track_model, newdata = df_base_validation)) %>%  # First predict for track_min_dist from regions
  mutate(
    wind_max_pred = predict(base_wind_model, newdata = df_base_validation),
    rain_total_pred = predict(base_rain_model, newdata = df_base_validation)
  ) %>% # recalculate interactions
  mutate(
      wind_blue_ss = wind_max_pred * blue_ss_frac,
      wind_yellow_ss = wind_max_pred * yellow_ss_frac,
      wind_orange_ss = wind_max_pred * orange_ss_frac,
      wind_red_ss = wind_max_pred * red_ss_frac,
      rain_blue_ss = rain_total_pred * blue_ls_frac,
      rain_yellow_ss = rain_total_pred * yellow_ls_frac,
      rain_orange_ss = rain_total_pred * orange_ls_frac,
      rain_red_ss = rain_total_pred * red_ls_frac,
  )


# Train the model using manual grid search
grid_id <- 1  # Index for list storage

# Iterate over all combinations of hyperparameters
for (i in 1:nrow(tune_grid)) {
  params <- tune_grid[i, ]
        # Train the model with specific hyperparameters
        xgb_model <- train(
          as.factor(damage_binary) ~ wind_max_pred +
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
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = params # Hyperparameter grid
        )

        # Make probability predictions for classification
        val_predictions <- predict(xgb_model, newdata = df_val_base_tune, type = "prob")[,2]  # Probability of class 1

        # Compute AUC (better for classification)
        auc_value <- auc(df_val_base_tune$damage_binary, val_predictions)

        # Store results efficiently in a list
        results_list[[i]] <- data.frame(params, AUC = auc_value)
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


damage_fit_class_min <- xgb_model <- train(
          as.factor(damage_binary) ~ wind_max_pred +
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
          data = final_training_df, # USE TRAINING AND VALIDATION SETS COMBINED
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = params # USE BEST PARAMETER
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
# Logging the model and parameter using MLflow

# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment 
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "Classify above 10 damage (Training metircs)")

# Start MLflow run
mlflow_start_run(nested = FALSE)

# Ensure the run ends even if an error occurs
on.exit(mlflow_end_run(), add = TRUE)

# Training
training_df$type <- as.factor(training_df$type)
model <- randomForest(type~ ., data = training_df, ntree = 100, proximity = TRUE)

# testing
preds  <- predict(model, test_df)

# summarize results
cnfm  <- confusionMatrix(preds, as.factor(test_df$type))

accuracy  <- cnfm$overall['Accuracy']

# Log parameters and metrics
mlflow_log_param("model_type", "random forest")
mlflow_log_metric("accuracy", accuracy)


# Save model
saveRDS(model, file = file.path(path_2_folder, "spam_clas_model.rds"))

#mlflow_log_artifact(path = file.path(path_2_folder, "spam_clas_model.rds"))
#saveRDS(model, file.path(output_dir, "model.rds"))

# Save model in MLflow format
#mlflow_save_model(model, path = paste0(path_2_folder, "mlflow_spam_clas_model"))

# End MLflow run
mlflow_end_run()

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

saveRDS(base_rain_model, file = paste0(managed_folder_path, "/base_rain_model.rds"))
saveRDS(base_track_model, file = paste0(managed_folder_path, "/base_track_model.rds"))