library(dataiku)

# Recipe inputs
modeling_data <- dkuReadDataset("modeling_data", samplingMethod="head", nbRows=100000)

# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
base_train <- replace_me # Compute a data frame for the output to write into base_train
base_test <- replace_me # Compute a data frame for the output to write into base_test
base_validation <- replace_me # Compute a data frame for the output to write into base_validation
truncated_train <- replace_me # Compute a data frame for the output to write into truncated_train
truncated_validation <- replace_me # Compute a data frame for the output to write into truncated_validation
truncated_test <- replace_me # Compute a data frame for the output to write into truncated_test


# Recipe outputs
dkuWriteDataset(base_train,"base_train")
dkuWriteDataset(base_test,"base_test")
dkuWriteDataset(base_validation,"base_validation")
dkuWriteDataset(truncated_train,"truncated_train")
dkuWriteDataset(truncated_validation,"truncated_validation")
dkuWriteDataset(truncated_test,"truncated_test")
