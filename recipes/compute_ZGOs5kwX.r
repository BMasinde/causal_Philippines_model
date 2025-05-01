library(dataiku)

# Recipe inputs
truncated_train <- dkuReadDataset("truncated_train", samplingMethod="head", nbRows=100000)
truncated_validation <- dkuReadDataset("truncated_validation", samplingMethod="head", nbRows=100000)
truncated_test <- dkuReadDataset("truncated_test", samplingMethod="head", nbRows=100000)



# Recipe outputs
ass_XGBOOST_trunc_reg <- dkuManagedFolderPath("ZGOs5kwX")
