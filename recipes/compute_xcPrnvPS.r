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
library(Matrix)
library(purrr) # useful for code optimization
library(themis)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe inputs
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
df_base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Combining train and validation datasets to one
# Because we are going to use CV to train the models later
# naming it df_base_train2 to remain consistent with df naming
df_base_train2  <- rbind(df_base_train, df_base_validation)

cat("number of rows in combined train data:", nrow(df_base_train2), sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training track_min_dist ~ island_groups
# we will need to also include island_groups
# in the final outcome prediction model to adjust for the confounding

base_track_model  <- rpart(track_min_dist  ~ island_groups,
                          data = df_base_train2,
                          method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)


base_wind_model <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train2,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for rain speed
# rain_total = f(track_min_dist, eps)

base_rain_model <- rpart(rain_total ~ track_min_dist,
                       data = df_base_train2,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Building typologies are determined by region
base_roof_strong_wall_strong_model  <- rpart(roof_strong_wall_strong  ~ island_groups, 
                                             data = df_base_train2,
                                            method = "anova")

base_roof_strong_wall_light_model  <- rpart(roof_strong_wall_light ~ island_groups,
                                           data = df_base_train2,
                                           method = "anova")

base_roof_strong_wall_salv_model  <- rpart(roof_strong_wall_salv ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")
base_roof_light_wall_strong_model  <- rpart(roof_light_wall_strong ~ island_groups,
                                           data = df_base_train2,
                                           method = "anova")
base_roof_light_wall_light_model  <- rpart(roof_light_wall_light ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")
base_roof_light_wall_salv_model  <- rpart(roof_light_wall_salv ~ island_groups,
                                         data = df_base_train2,
                                         method = "anova")

base_roof_salv_wall_strong_model  <- rpart(roof_salv_wall_strong ~ island_groups,
                                          data = df_base_train2,
                                          method = "anova")

base_roof_salv_wall_light_model  <- rpart(roof_salv_wall_light ~ island_groups,
                                  data = df_base_train2,
                                  method = "anova")

base_roof_salv_wall_salv_model  <- rpart(roof_salv_wall_salv ~ island_groups,
                                  data = df_base_train2,
                                  method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# OPTMIZED CODE
# Define models in a named list
model_list <- list(
  track_min_dist = base_track_model,
  wind_max = base_wind_model,
  rain_total = base_rain_model,
  roof_strong_wall_strong = base_roof_strong_wall_strong_model,
  roof_strong_wall_light = base_roof_strong_wall_light_model,
  roof_strong_wall_salv = base_roof_strong_wall_salv_model,
  roof_light_wall_strong = base_roof_light_wall_strong_model,
  roof_light_wall_light = base_roof_light_wall_light_model,
  roof_light_wall_salv = base_roof_light_wall_salv_model,
  roof_salv_wall_strong = base_roof_salv_wall_strong_model,
  roof_salv_wall_light = base_roof_salv_wall_light_model,
  roof_salv_wall_salv = base_roof_salv_wall_salv_model
)

# Apply predictions efficiently
df_base_train2 <- df_base_train2 %>%
  mutate(across(names(model_list), ~ predict(model_list[[cur_column()]], newdata = df_base_train2), .names = "{.col}_pred")) 

# Define wind and rain interaction variables
wind_fractions <- c("blue_ss_frac", "yellow_ss_frac", "orange_ss_frac", "red_ss_frac")
rain_fractions <- c("blue_ls_frac", "yellow_ls_frac", "orange_ls_frac", "red_ls_frac")

# Compute wind interaction terms dynamically
df_base_train2 <- df_base_train2 %>%
  mutate(across(all_of(wind_fractions), ~ . * wind_max_pred, .names = "wind_{.col}"),
         across(all_of(rain_fractions), ~ . * rain_total_pred, .names = "rain_{.col}"))
# ------------------------------- OLD MODEL TRAINING
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adding the predicted parents' to the training dataset

# df_train <- df_train %>%
#   mutate(track_min_dist_pred = predict(base_track_model, newdata = df_base_train2), # predicted min_dist
#          wind_max_pred = predict(base_wind_model, newdata = df_base_train2),
#          rain_total_pred = predict(base_rain_model, newdata = df_base_train2), 
#          #---- Updating interaction terms ------------------------
#          wind_blue_ss = wind_max_pred * blue_ss_frac,
#          wind_yellow_ss = wind_max_pred * yellow_ss_frac,
#          wind_orange_ss = wind_max_pred * orange_ss_frac,
#          wind_red_ss = wind_max_pred * red_ss_frac,
#          rain_blue_ss = rain_total_pred * blue_ls_frac,
#          rain_yellow_ss = rain_total_pred * yellow_ls_frac,
#          rain_orange_ss = rain_total_pred * orange_ls_frac,
#          rain_red_ss = rain_total_pred * red_ls_frac,
#          # -------- Updating building typologies ------------------
#          roof_strong_wall_strong_pred = predict(base_roof_strong_wall_strong_model, newdata = df_base_train2), 
#          roof_strong_wall_light_pred = predict(base_roof_strong_wall_light_model, newdata = df_base_train2),
#          roof_strong_wall_salv_pred = predict(base_roof_strong_wall_salv_model, newdata = df_base_train2),
#          roof_light_wall_strong_pred = predict(base_roof_light_wall_strong_model, newdata = df_base_train2),
#          roof_light_wall_light_pred = predict(base_roof_light_wall_light_model, newdata = df_base_train2),
#          roof_light_wall_salv_pred = predict(base_roof_light_wall_salv_model, newdata = df_base_train2),
#          roof_salv_wall_strong_pred = predict(base_roof_salv_wall_strong_model, newdata = df_base_train2),
#          roof_salv_wall_light_pred = predict(base_roof_salv_wall_light_model, newdata = df_base_train2),
#          roof_salv_wall_salv_pred = predict(base_roof_salv_wall_salv_model, newdata = df_base_train2), 
#          )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#--------------------------- NOT NEEDED BECAUSE WE OPT TO DO CV IN TRAINING -------------------------------------
# Adding the predicted parents' to the validation dataset
# predicting for wind and rainfall for the validation dataset
#df_base_validation <- df_base_validation %>%
#  mutate(track_min_dist_pred = predict(base_track_model, newdata = df_base_validation),  # First predict for track_min_dist from regions
#    wind_max_pred = predict(base_wind_model, newdata = df_base_validation),
#    rain_total_pred = predict(base_rain_model, newdata = df_base_validation),
#    wind_blue_ss = wind_max_pred * blue_ss_frac,
#    wind_yellow_ss = wind_max_pred * yellow_ss_frac,
#    wind_orange_ss = wind_max_pred * orange_ss_frac,
#    wind_red_ss = wind_max_pred * red_ss_frac,
#    rain_blue_ss = rain_total_pred * blue_ls_frac,
#    rain_yellow_ss = rain_total_pred * yellow_ls_frac,
#    rain_orange_ss = rain_total_pred * orange_ls_frac,
#    rain_red_ss = rain_total_pred * red_ls_frac,
#  )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ------------------ GRID SEARCH TUNING ------------------------------------
# # Parameter tuning

# # Define tuning grid
# tune_grid <- expand.grid(
#   nrounds = c(50, 100, 150),       # Number of boosting rounds
#   max_depth = c(3, 6, 9),          # Maximum tree depth
#   eta = c(0.01, 0.1, 0.3),         # Learning rate
#   gamma = 0,                       # Minimum loss reduction
#   colsample_bytree = 0.8,          # Feature selection rate
#   min_child_weight = 1,            # Minimum instance weight
#   subsample = 0.8                  # Sample ratio per boosting round
# )


# # Create an empty list to store results
# results_list <- list()

# # Extra data prep
# # Ensure target variable is a factor for classification
# df_base_train2$damage_binary <- as.factor(df_base_train2$damage_binary)
# #df_base_validation$damage_binary <- as.factor(df_base_validation$damage_binary)

# # Train the model using manual grid search
# grid_id <- 1  # Index for list storage

# # Iterate over all combinations of hyperparameters
# for (i in 1:nrow(tune_grid)) {
#   params <- tune_grid[i, ]

#         # setting seed for reproducibility
#         set.seed(1234)
#         # Train the model with specific hyperparameters
#         xgb_model <- train(
#           as.factor(damage_binary) ~ wind_max_pred +
#             rain_total_pred +
#             roof_strong_wall_strong +
#             roof_strong_wall_light +
#             roof_strong_wall_salv +
#             roof_light_wall_strong +
#             roof_light_wall_light +
#             roof_light_wall_salv +
#             roof_salv_wall_strong +
#             roof_salv_wall_light +
#             roof_salv_wall_salv +
#             ls_risk_pct +
#             ss_risk_pct +
#             wind_blue_ss +
#             wind_yellow_ss +
#             wind_orange_ss +
#             wind_red_ss +
#             rain_blue_ss +
#             rain_yellow_ss +
#             rain_orange_ss +
#             rain_red_ss +
#             island_groups, # CONFOUNDER ADJUSTED
#           data = df_base_train,
#           method = "xgbTree", # XGBoost method
#           trControl = trainControl(method = "none"),  # No automatic validation
#           tuneGrid = params # Hyperparameter grid
#         )

#         # Make probability predictions for classification
#         val_predictions <- predict(xgb_model, newdata = df_base_validation, type = "prob")[,2]  # Probability of class 1

#         # Compute AUC (better for classification)
#         auc_value <- auc(df_base_validation$damage_binary, val_predictions)

#         # Store results efficiently in a list
#         results_list[[i]] <- data.frame(params, AUC = auc_value)
# }

# # Convert list to data frame
# results <- rbindlist(results_list)

# # Print the best hyperparameter combination (highest AUC)
# best_params <- results[which.max(results$AUC), ]
# print(best_params)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Ensure target variable is a factor
# Ensure the target variable is a factor with valid names

#df_base_train2$damage_binary <- as.factor(df_base_train2$damage_binary)

# -------------------------------------------------------------------------------------------------------------

df_base_train2$damage_binary_2 <- factor(df_base_train2$damage_binary, 
                                       levels = c("0", "1"),  # Your current levels
                                       labels = c("Damage_below_10", "Damage_above_10"))  # New valid labels

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ---------------------- CLASSIFICATION MDOEL TRAINING WITH 10 CV & GRID SEARCH PARAMETER TUNING -----------------
# Define tuning grid
# tune_grid <- expand.grid(
#   nrounds = c(50, 100, 200, 300, 400, 500),
#   max_depth = c(3, 6, 9, 12),
#   eta = c(0.01, 0.05, 0.1, 0.2, 0.3),
#   gamma = c(0, 1, 5, 10),
#   colsample_bytree = c(0.5, 0.7, 0.8, 1.0),
#   min_child_weight = c(1, 3, 5, 10),
#   subsample = c(0.5, 0.7, 0.8, 1.0)
# )

tune_grid <- expand.grid(
  nrounds = c(50, 100, 200),
  max_depth = c(3, 6),
  eta = c(0.1, 0.2),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 1.0)
)


# Set up train control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,  # Needed for AUC calculation
  summaryFunction = twoClassSummary,
  sampling = "smote" # caret automatically identifies minority class
)

# Measure the time for a code block to run
system.time({
    # Train the model using grid search with 10-fold CV
    set.seed(1234)
    xgb_model <- train(
      damage_binary_2 ~ wind_max_pred +
        rain_total_pred +
        roof_strong_wall_strong_pred +
        roof_strong_wall_light_pred +
        roof_strong_wall_salv_pred +
        roof_light_wall_strong_pred +
        roof_light_wall_light_pred +
        roof_light_wall_salv_pred +
        roof_salv_wall_strong_pred +
        roof_salv_wall_light_pred +
        roof_salv_wall_salv_pred +
        ls_risk_pct +
        ss_risk_pct +
        wind_blue_ss +
        wind_yellow_ss +
        wind_orange_ss +
        wind_red_ss +
        rain_blue_ss +
        rain_yellow_ss +
        rain_orange_ss +
        rain_red_ss +
        island_groups +  # Confounder adjustment
        track_min_dist_pred, # Confounder adjustment
        data = df_base_train2,
        method = "xgbTree",
        trControl = train_control,
        tuneGrid = tune_grid,
        metric = "ROC", # Optimize based on AUC
        sample = "smote"
    )
    Sys.sleep(2)  # This is just an example to simulate a delay
})
    
# Print best parameters
print(xgb_model$bestTune)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
xgb_model$bestTune

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training based on tuned parameters

# Combine Training and Validation datasets for final training

#final_training_df  <- rbind(df_base_train,
#                           df_base_validation)


# Extract the best parameters (remove AUC column)
best_params_model <- xgb_model$bestTune

damage_fit_class_full <- train(
          damage_binary_2 ~ wind_max_pred +
            rain_total_pred +
            roof_strong_wall_strong_pred +
            roof_strong_wall_light_pred +
            roof_strong_wall_salv_pred +
            roof_light_wall_strong_pred +
            roof_light_wall_light_pred +
            roof_light_wall_salv_pred +
            roof_salv_wall_strong_pred +
            roof_salv_wall_light_pred +
            roof_salv_wall_salv_pred +
            ls_risk_pct +
            ss_risk_pct +
            wind_blue_ss +
            wind_yellow_ss +
            wind_orange_ss +
            wind_red_ss +
            rain_blue_ss +
            rain_yellow_ss +
            rain_orange_ss +
            rain_red_ss +
            island_groups +  # Confounder adjustment
           track_min_dist_pred, # Confounder adjustment
          data = df_base_train2, # USE TRAINING AND VALIDATION SETS COMBINED
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = best_params_model # USE BEST PARAMETER
        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Sanity Check
# testing on the training datasets (training + validation)

## Outcome prediction on the final_training_df dataset
## default function predict returns class probabilities (has two columns)
y_pred <- predict(damage_fit_class_full,
                  newdata = df_base_train2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
levels(y_pred)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# using table function
conf_matrix <- confusionMatrix(y_pred,
                     df_base_train2$damage_binary_2, # remember to use damage_binary_2
                     positive = "Damage_above_10"
                     )
conf_matrix

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
accuracy <- conf_matrix$overall['Accuracy']

cat("test-set accuracy of minimal SCM model:", accuracy, sep = " ")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Logging the model and parameter using MLflow

# set tracking URI
mlflow_set_tracking_uri("http://127.0.0.1:5000")

# Ensure any active run is ended
suppressWarnings(try(mlflow_end_run(), silent = TRUE))

# set experiment
# Logging metrics for model training and the parameters used
mlflow_set_experiment(experiment_name = "SCM - XGBOOST classification - CV (Training metircs)")

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
mlflow_log_param("model_type", "scm-xgboost-classification")

damage_fit_class_full <- train(
          damage_binary_2 ~ wind_max_pred +
            rain_total_pred +
            roof_strong_wall_strong_pred +
            roof_strong_wall_light_pred +
            roof_strong_wall_salv_pred +
            roof_light_wall_strong_pred +
            roof_light_wall_light_pred +
            roof_light_wall_salv_pred +
            roof_salv_wall_strong_pred +
            roof_salv_wall_light_pred +
            roof_salv_wall_salv_pred +
            ls_risk_pct +
            ss_risk_pct +
            wind_blue_ss +
            wind_yellow_ss +
            wind_orange_ss +
            wind_red_ss +
            rain_blue_ss +
            rain_yellow_ss +
            rain_orange_ss +
            rain_red_ss +
            island_groups +  # Confounder adjustment
           track_min_dist_pred, # Confounder adjustment
          data = final_training_df, # USE TRAINING AND VALIDATION SETS COMBINED
          method = "xgbTree", # XGBoost method
          trControl = trainControl(method = "none"),  # No automatic validation
          tuneGrid = best_params_model # USE BEST PARAMETER
        )


# summarize results
conf_matrix <- confusionMatrix(y_pred,
                     final_training_df$damage_binary_2,
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
# Recipe outputs
managed_folder_path <- dkuManagedFolderPath("xcPrnvPS")

# ------------ Models in a list -----------------------
models <- list(damage_fit_class_full,
               base_wind_model, 
               base_rain_model, 
               base_track_model,
               base_roof_strong_wall_strong_model,
               base_roof_strong_wall_light_model,
               base_roof_strong_wall_salv_model,
               base_roof_light_wall_strong_model,
               base_roof_light_wall_light_model,
               base_roof_light_wall_salv_model, 
               base_roof_salv_wall_strong_model,
               base_roof_salv_wall_light_model,
               base_roof_salv_wall_salv_model
              )
model_names <- c("base_clas_full_model",
                 "base_wind_model", 
                 "base_rain_model", 
                 "base_track_model",
                 "base_roof_strong_wall_strong_model",
                 "base_roof_strong_wall_light_model",
                 "base_roof_strong_wall_salv_model",
                 "base_roof_light_wall_strong_model",
                 "base_roof_light_wall_light_model",
                 "base_roof_light_wall_salv_model", 
                 "base_roof_salv_wall_strong_model",
                 "base_roof_salv_wall_light_model",
                 "base_roof_salv_wall_salv_model"
                )

#----------------------- Saving trained XGBOOST model ----------------------------------------
mapply(function(model, name) {
  saveRDS(model, file = paste0(managed_folder_path, "/", name, ".rds"))
}, models, model_names)

#----------------------- Saving trained XGBOOST model ----------------------------------------
#saveRDS(damage_fit_class_full, file = paste0(managed_folder_path, "/base_clas_full_model.rds"))

#----------------------- Saving parent node models for hazard vars ----------------------------
#saveRDS(base_wind_model, file = paste0(managed_folder_path, "/base_wind_model.rds"))
#saveRDS(base_rain_model, file = paste0(managed_folder_path, "/base_rain_model.rds"))
#saveRDS(base_track_model, file = paste0(managed_folder_path, "/base_track_model.rds"))

#----------------------- Saving parent node models for hazard vars -----------------------------