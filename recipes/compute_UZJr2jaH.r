# Load required libraries
library(dataiku)
library(mlflow)

# Set up Dataiku Managed Folders
scm_models <- dkuManagedFolderPath("XxDUuuYe")  # Input Managed Folder
mlflow_output <- dkuManagedFolderPath("UZJr2jaH")  # Output Managed Folder

# Load trained R model from the Managed Folder
model_path <- file.path(scm_models, "base_wind_pred.rds")
model <- readRDS(model_path)

# Set MLflow Tracking URI (optional: customize if using external MLflow server)
mlflow_set_tracking_uri("file:///mlruns")  # Change this if using a remote MLflow server

# Start MLflow experiment
mlflow_set_experiment("wind_prediction_experiment")

# Start MLflow run
mlflow_start_run()

# Log model metadata (optional: add parameters or metrics)
mlflow_log_param("model_type", "random_forest")  # Example: specify model type
mlflow_log_metric("accuracy", 0.95)  # Example: log accuracy if available

# Log the model in MLflow
mlflow_rfunc_model(model, "wind_model")

# Save MLflow model to the Managed Folder
mlflow_model_path <- file.path(mlflow_output, "wind_model")
mlflow_save_model(model, mlflow_model_path)

# End the MLflow run
mlflow_end_run()

print("Model successfully logged in MLflow and saved to Dataiku Managed Folder.")
