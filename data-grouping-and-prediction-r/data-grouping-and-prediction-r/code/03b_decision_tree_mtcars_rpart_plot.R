library(rpart)
data(mtcars)

# Convert target to categorical
mtcars$am <- factor(mtcars$am)

# Build Decision Tree
dt <- rpart(am ~ mpg + hp + wt + cyl, data=mtcars)

# Print tree structure
print(dt)

# Visualize Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(dt)
