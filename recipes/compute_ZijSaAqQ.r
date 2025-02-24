library(dataiku)

# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)



# Recipe outputs
base_scm_reg_min_model <- dkuManagedFolderPath("ZijSaAqQ")
