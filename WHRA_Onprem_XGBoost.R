#
#Most Successful solution for the Womens Health Risk Assessment competition by Cortana Intelligence
#https://gallery.cortanaintelligence.com/Competition/Women-s-Health-Risk-Assessment-1
#Ion Kleopas
#

# Install the xgboost package if it is not already installed
install.packages('xgboost')
# Load the library
library(xgboost)
# Load the dataset
dataURL <- 'http://az754797.vo.msecnd.net/competition/whra/data/WomenHealth_Training.csv'

# Load the dataset into an R table, with the correct format for each cell
colclasses <- rep("integer",50)
colclasses[36] <- "character"
dataset1 <- read.table(dataURL, header=TRUE, sep=",", strip.white=TRUE, stringsAsFactors = F, colClasses = colclasses)

# Combine columns geo, segment, and subgroup into a single column. 
# This will be the label column to be predicted in the multiclass classification task
combined_label <- 100*dataset1$geo + 10*dataset1$segment + dataset1$subgroup
data.set <- cbind(dataset1, combined_label)

# Make sure that data.set has the correct data types
for(i in 1:ncol(data.set)){
  if (i!=36) {
    data.set[, i] <- as.integer(data.set[, i])
  }
}
data.set[, 36] <- as.character(data.set[, 36])

data.set$combined_label <- as.factor(data.set$combined_label)

# Cleaning missing data - replace NAs and empty characters
data.set[is.na(data.set)] <- 0
data.set[data.set$religion=="", "religion"] <- "0"
data.set$religion <- factor(data.set$religion)
data.set$combined_label <- relevel(data.set$combined_label, ref = '111')

# Split the data into train (75%) and validation data (25%)
nrows <- nrow(data.set)
sample_size <- floor(0.75 * nrows)
set.seed(98052) # set the seed to make the partition reproductible
train_ind <- sample(seq_len(nrows), size = sample_size)

train <- data.set[train_ind, ]
validation <- data.set[-train_ind, ]

###Fix and prepare the factors 
lab <- train$combined_label
lab.lev <- levels(lab)
lab <- as.numeric(lab)
train[,52] <- lab

##Train an XGBoost model with the training data
xgb11 <- xgboost(xgb.DMatrix(data.matrix(train[-c(1,19,49,50,51,52)]),label=t(train[52]), missing = NaN), max.depth = 20, eta = 0.025, nround = 1000,
                 objective = "multi:softmax", num_class = 38, gamma = 1)

# Predict - evaluate the performance of the trained model on the remaining portion of the split data.
pred <- predict(xgb11, xgb.DMatrix(data.matrix(validation[-c(1,19,49,50,51,52)]),label=t(rep(0,nrow(validation))), missing = NaN))
pred <- lab.lev[pred]
accuracy <- round(sum(pred==validation$combined_label)/nrow(validation) * 100,6)

# Print the accuracy
print(paste("The accuracy on validation data is ", accuracy, "%", sep=""))

##
##Once we reach a satisfying accuracy, train an XGBoost model using the whole dataset intead of the 75% slpit.
lab <- data.set$combined_label
lab.lev <- levels(lab)
lab <- as.numeric(lab)
data.set[,52] <- lab

# Train using the whole dataset
xgb11 <- xgboost(xgb.DMatrix(data.matrix(data.set[-c(1,19,49,50,51,52)]),label=t(data.set[52]), missing = NaN), max.depth = 20, eta = 0.025, nround = 1000,
                 objective = "multi:softmax", num_class = 38, gamma = 1)

# Save our model
save(xgb11,file="xgb11.rda")
