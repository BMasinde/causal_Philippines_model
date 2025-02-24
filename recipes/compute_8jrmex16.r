library(dataiku)

# Recipe inputs
base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)



# Recipe outputs
base_scm_classification_model <- dkuManagedFolderPath("8jrmex16")
