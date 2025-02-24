library(dataiku)

# Recipe inputs
base_scm_clas_min_model <- dkuManagedFolderPath("xcPrnvPS")
base_test <- dkuReadDataset("base_test", samplingMethod="head", nbRows=100000)



# Recipe outputs
scm_clas_min_model_test <- dkuManagedFolderPath("Xu27U2QF")
