library(dataiku)

# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)



# Recipe outputs
ass_XGBOOST_classifier <- dkuManagedFolderPath("fZ8zhmA4")
