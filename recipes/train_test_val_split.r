# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Train, Test, and Validation Dataset Creation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# We do a 60/20/20 split to create the datasets. Because we are going to use the hurdle method for predictions we split the modeling data into two categories: base and truncated. No additional processing (filtering) is required to create "base_" datasets. To create "truncated_" datasets we filter the modeling data by outcome variable (damage_perc >= 10).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Libraries
library(dataiku)
library(dplyr)

# Recipe inputs
modeling_data <- dkuReadDataset("modeling_data", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# splits for base_ datasets

# number of rows in modeling_data
n <- nrow(modeling_data)

# Seeding for reproducibility
set.seed(12345)

# Generate random indices for 60% training set
base_train_id <- sample(1:n, floor(n * 0.6), replace = FALSE)

# Remaining indices after training selection
base_remaining_id <- setdiff(1:n, base_train_id)

# Split remaining 40% into 20% validation and 20% test
base_val_id <- sample(base_remaining_id, floor(n * 0.2))

base_test_id <- setdiff(base_remaining_id, base_val_id)  # The rest goes to test


# Compute recipe outputs for base_ datasets
base_train <- modeling_data[base_train_id, ] # Compute a data frame for the output to write into base_train

base_test <- modeling_data[base_test_id, ] # Compute a data frame for the output to write into base_test

base_validation <- modeling_data[base_val_id, ] # Compute a data frame for the output to write into base_validation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# splits for truncated_ datasets

# Filtering modeling data by column damage_perc >= 10
truncated_data <- modeling_data %>%
  filter(damage_perc >= 10.0)

# Reset row ID's
rownames(truncated_data) <- 1:nrow(truncated_data)

# number of observations with damage > 10
n_trunc <- nrow(truncated_data)

# Sample 60% for training
trunc_train_id <- sample(1:n_trunc, floor(n_trunc * 0.6), replace = FALSE)

# Get remaining 40% indices
trunc_remaining_id <- setdiff(1:n_trunc, trunc_train_id)

# Calculate correct split for validation and test (each should be 50% of the remaining)
n_remaining <- length(trunc_remaining_id)
val_size <- floor(n_remaining * 0.5)  # 50% of remaining

# Sample validation set from remaining
trunc_val_id <- sample(trunc_remaining_id, val_size, replace = FALSE)

## The rest (remaining 20%) goes to test
trunc_test_id <- setdiff(trunc_remaining_id, trunc_val_id)


# Compute recipe outputs for truncated_ datasets
truncated_train <- truncated_data[trunc_train_id, ] # Compute a data frame for the output to write into truncated_train

truncated_validation <- truncated_data[trunc_val_id, ] # Compute a data frame for the output to write into truncated_validation

truncated_test <- truncated_data[trunc_test_id, ] # Compute a data frame for the output to write into truncated_test

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
dkuWriteDataset(base_train,"base_train")
dkuWriteDataset(base_test,"base_test")
dkuWriteDataset(base_validation,"base_validation")
dkuWriteDataset(truncated_train,"truncated_train")
dkuWriteDataset(truncated_validation,"truncated_validation")
dkuWriteDataset(truncated_test,"truncated_test")