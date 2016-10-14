#dataset1 is the web service input: The dataset to be evaluated. (training on the training set was done locally in R Studio)
dataset1 <- maml.mapInputPort(1)
#dataset2 contains the 38 possible classes, in a particular order, to help with mapping back the results of the classification 
#to the original geo+segment+subgroup class labels. (eg to map class 1 to 111, class 38 to 932, etc.)  
dataset2 <- maml.mapInputPort(2)

###Convert to the correct data types, all integers except for column 36
for(i in 1:ncol(dataset1)){
  if (i!=36) {
    dataset1[, i] <- as.integer(dataset1[, i])
  }
}
dataset1[, 36] <- as.character(dataset1[, 36])

###Clean the data
#Replace NA's of each column with the mean 
for(i in 1:ncol(dataset1)){
  if (i!=36) {
    dataset1[is.na(dataset1[,i]), i] <- mean(dataset1[,i], na.rm = TRUE)
  }
}
dataset1[is.na(dataset1)] <- 0
religion <- dataset1$religion
religion[religion==""] <- "0"
religion[is.na(religion)] <- "0"
dataset1$religion <- factor(religion)

#Install and load the required R packages from the xgbee.zip file.
#The packages had to be uploaded manually in a zip file along with the exported .rda model 
#since I couldn't find another way to install them in ML Studio. 
install.packages("src/magrittr.zip", lib = ".", repos = NULL, verbose = TRUE)
install.packages("src/xgboost.zip", lib = ".", repos = NULL, verbose = TRUE)
library(magrittr,lib.loc = ".")
library(xgboost,lib.loc = ".")

#Load the exported .rda model from the zip file
#This is an XGBoost model built locally in R Studio
load('src/xgb11.rda') 
#Predict the labels of the dataset to be evaluated, using the model we just loaded. 
#Note that XGBoost's input must be a matrix. 
predicted_labels <- predict(xgb11, xgb.DMatrix(data.matrix(dataset1[-c(1,19,49,50)])), missing = NaN)

#Use lab.lev.csv (dataset2) to map the predicted classes back to the original labels.
#It contains the 38 possible class labels, in a particular order, to help with mapping back the results of the classification 
#to the original geo+segment+subgroup class labels. (eg to map class 1 to 111, class 38 to 932, etc.)
#This was achieved by considering the position of each geo-segment-subgroup label in lab.lev.csv as its corresponding class
#In other words, if class '932' is in the 38 position of the array, it corresponds to class 38 of the model.  
predicted_labels <- dataset2$LEVEL[predicted_labels]
nrows <- length(predicted_labels)
predictions <- matrix(rep(0,nrows*3), nrow=nrows, ncol=3)
for (i in 1:nrows){
  x <- as.character(predicted_labels[i])
  predictions[i,] <- as.numeric(unlist(strsplit(x, "")))
}
#Map to geo+segment+subgroup
predictions <- as.data.frame(predictions)
data.set <- data.frame(dataset1[,1], predictions)
colnames(data.set) <- c("patientID", "Geo_Pred","Segment_Pred", "Subgroup_Pred")

#Output the predicted dataset.
maml.mapOutputPort("data.set");

