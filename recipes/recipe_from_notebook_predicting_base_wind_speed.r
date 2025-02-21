# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
library(dataiku)
library(rpart)
#library(mlflow)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the dataset as a R dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Training structural equation for wind speed
# wind_speed = f(track_min_dist, eps)

base_wind_max_model <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the trained model in a Managed Folder
dkuManagedFolderPath <- dkuManagedFolderPath("scm_models")
saveRDS(base_wind_max_model, file = paste0(dkuManagedFolderPath, "/base_wind_max_model.rds"))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the trained model in a Managed Folder

# Start MLflow run
#mlflow_start_run()

# Save model locally (as a temporary directory or file)
#model_path <- tempfile("mlflow_model_")
#mlflow_save_model(model, model_path)

# Access the managed folder in Dataiku DSS
#managed_folder <- dataiku::managedFolder("scm_models")

# Define the path inside the managed folder
#managed_folder_path <- file.path(managed_folder$getPath(), "base_wind_max_model")

# Copy the saved model to the managed folder
#file.copy(model_path, managed_folder_path, recursive = TRUE)

# Log the parameters and metrics (optional)
#mlflow_log_param("model_type", "Regression Tree")
#mlflow_log_metric("RMSE", 0.95)