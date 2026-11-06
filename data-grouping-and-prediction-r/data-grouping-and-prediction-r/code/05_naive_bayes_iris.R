
# Load library
library(e1071)
# Sample dataset
data(iris)
# Split dataset
set.seed(123)
index <- sample(1:nrow(iris), 0.7*nrow(iris))
train <- iris[index,]
test <- iris[-index,]
# Train Naive Bayes model
model <- naiveBayes(Species ~ ., data=train)
# Prediction
pred <- predict(model, test)
# Accuracy
table(pred, test$Species)
mean(pred == test$Species)



