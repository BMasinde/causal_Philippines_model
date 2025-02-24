library(dataiku)

# Recipe inputs
scm_base_models <- dkuManagedFolderPath("XxDUuuYe")
base_validation <- dkuReadDataset("base_validation", samplingMethod="head", nbRows=100000)



# Recipe outputs
base_val_scm_models <- dkuManagedFolderPath("ZMqSlX9h")
