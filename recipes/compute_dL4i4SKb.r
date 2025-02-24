library(dataiku)

# Recipe inputs
truncated_train <- dkuReadDataset("truncated_train", samplingMethod="head", nbRows=100000)
truncated_validation <- dkuReadDataset("truncated_validation", samplingMethod="head", nbRows=100000)



# Recipe outputs
trunk_scm_min_model <- dkuManagedFolderPath("dL4i4SKb")
