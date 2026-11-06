rm(list = ls())
data(iris)

# ── ALGORITHM 1: LOGISTIC REGRESSION ────────────────────────
iris$y_num <- ifelse(iris$Species == "setosa", 1, 0)
m1 <- glm(y_num ~ Sepal.Length + Sepal.Width,
          data   = iris,
          family = "binomial")
p1 <- ifelse(predict(m1, type = "response") > 0.5, 1, 0)
acc_log <- mean(p1 == iris$y_num)
cat("Logistic Regression Accuracy:", round(acc_log * 100, 1), "%\n")
plot(iris$Sepal.Length, iris$Sepal.Width,
     col=p1+1, pch=19,
     main="Logistic Regression",
     xlab="Sepal Length", ylab="Sepal Width")
legend("topright", legend=c("Not Setosa","Setosa"),
       col=c(1,2), pch=19)

# ── ALGORITHM 2: KNN ─────────────────────────────────────────
library(class)
x     <- scale(iris[, 1:2])
y_num <- ifelse(iris$Species == "setosa", 1, 0)
p2 <- knn(train=x, test=x, cl=y_num, k=5)
acc_knn <- mean(as.numeric(as.character(p2)) == y_num)
cat("KNN Accuracy                :", round(acc_knn * 100, 1), "%\n")
plot(iris$Sepal.Length, iris$Sepal.Width,
     col=as.numeric(as.character(p2))+1, pch=19,
     main="KNN Classification (k=5)",
     xlab="Sepal Length", ylab="Sepal Width")
legend("topright", legend=c("Not Setosa","Setosa"),
       col=c(1,2), pch=19)

# ── ALGORITHM 3: DECISION TREE ───────────────────────────────
library(rpart)
iris$y_fac <- factor(ifelse(iris$Species == "setosa", "Yes", "No"))
m3 <- rpart(y_fac ~ Sepal.Length + Sepal.Width, data=iris)
p3 <- predict(m3, type="class")
acc_tree <- mean(p3 == iris$y_fac)
cat("Decision Tree Accuracy      :", round(acc_tree * 100, 1), "%\n")
plot(iris$Sepal.Length, iris$Sepal.Width,
     col=as.numeric(p3)+1, pch=19,
     main="Decision Tree Classification",
     xlab="Sepal Length", ylab="Sepal Width")
legend("topright", legend=c("Not Setosa","Setosa"),
       col=c(2,3), pch=19)

# ── COMPARISON BAR PLOT ──────────────────────────────────────
acc <- c(Logistic = acc_log,
         KNN      = acc_knn,
         Tree     = acc_tree)

barplot(acc,
        col       = c("black","orange","yellow"),
        ylim      = c(0, 1),
        ylab      = "Accuracy",
        names.arg = c("Logistic","KNN","Tree"),
        axes      = FALSE,
        bty       = "n")       # ← removes the box border

axis(2, at     = c(0.0, 0.4, 0.8),
     labels = c("0.0","0.4","0.8"),
     las    = 0)            # ← rotates numbers vertically
