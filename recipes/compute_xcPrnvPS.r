library(dataiku)

# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)



# Recipe outputs
base_scm_clas_min_model <- dkuManagedFolderPath("xcPrnvPS")
