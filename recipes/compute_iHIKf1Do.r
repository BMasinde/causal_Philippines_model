library(dataiku)

# Recipe inputs
truncated_test <- dkuReadDataset("truncated_test", samplingMethod="head", nbRows=100000)
trunk_scm_min_model <- dkuManagedFolderPath("dL4i4SKb")



# Recipe outputs
trunk_scm_min_model_test <- dkuManagedFolderPath("iHIKf1Do")
