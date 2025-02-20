# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Model Predicting Effect of Track on Wind
# As determined in exploratory data analysis, the relationship between maximum sustained wind speed and the track distance between municipality and the tropical cyclone's path is non-linear. To capture this non-linearity, we use decision trees.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Libraries
library(dataiku)
library(rpart) # decision trees library

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the dataset as a R dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
df_base_train <- dkuReadDataset("base_train", samplingMethod="head", nbRows=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# training on base_train data
wind_max_fit <- rpart(wind_max ~ track_min_dist,
                       data = df_base_train,
                       method = "anova")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# scm prediction for wind
# wind_max = f(track_min_dist, e)
wind_max_pred <- predict(wind_max_fit,
                         newdata = df_base_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# creating a dataframe with the prediction
base_wind_pred  <- as.data.frame(wind_max_pred)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
dkuWriteDataset(base_wind_pred,"base_wind_pred")