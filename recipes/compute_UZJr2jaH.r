# Load required libraries
library(dataiku)
library(mlflow)

# Set up Dataiku Managed Folders
scm_models <- dkuManagedFolderPath("XxDUuuYe")  # Input Managed Folder
mlflow_output <- dkuManagedFolderPath("UZJr2jaH")  # Output Managed Folder

# Define path to download the RDS file (temporary location)
local_model_path <- "/tmp/base_wind_pred.rds"

# Download file from Dataiku Managed Folder
dkuManagedFolderDownloadPath("XxDUuuYe", "base_wind_pred.rds", local_model_path)

# Check if the file exists
if (!file.exists(local_model_path)) {
  stop("Error: The RDS file could not be downloaded from the Managed Folder.")
}

# Load the trained R model
model <- readRDS(local_model_path)

# Set MLflow Tracking URI (optional: customize if using external MLflow server)
mlflow_set_tracking_uri("file:///mlruns")  # Change this if using a remote MLflow server

# Start MLflow experiment
mlflow_set_experiment("base_wind_prediction_experiment")

# Start MLflow run
mlflow_start_run()

# Log model metadata (optional)
mlflow_log_param("model_type", "Regression Trees")  # Example: specify model type
mlflow_log_metric("RMSE",7.0000)  # Example: log accuracy if available

# Log the model in MLflow
mlflow_rfunc_model(model, "base_wind_model")

# Save MLflow model to the Managed Folder
mlflow_model_path <- file.path(mlflow_output, "wind_model")
mlflow_save_model(model, mlflow_model_path)

# End the MLflow run
mlflow_end_run()

print("Model successfully logged in MLflow and saved to Dataiku Managed Folder.")
