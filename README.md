# Data Grouping and Prediction – Sample Datasets

**College Project | MCA – Data Mining Using R Lab | Tool: R**

---

## 1. What is this project about?

This project uses R to do two common jobs in data analysis on small, well-known sample datasets:

1. **Grouping (clustering):** putting similar records together, without telling the computer the groups in advance. For example, grouping US states with similar crime figures.
2. **Predicting (classification):** teaching the computer from past data so it can predict a category. For example, guessing whether a car has an automatic or manual gearbox from its weight and mileage.

| Part | Job | Method | Dataset |
|---|---|---|---|
| A | Grouping | K-Means and Hierarchical Clustering | Iris (flowers) |
| B | Grouping | K-Means Clustering | USArrests (US states) |
| C | Predicting | Decision Tree | mtcars (cars) |
| D | Predicting | Naive Bayes | Iris (flowers) |
| E | Predicting | Logistic Regression, KNN and Decision Tree compared | Iris (flowers) |

---

## 2. About the data

All four datasets come **built into R**, so nothing needs to be downloaded.

| Dataset | What one row means | Size | What it contains |
|---|---|---|---|
| **iris** | One flower | 150 flowers | Sepal length, sepal width, petal length, petal width, and the species (Setosa, Versicolor, Virginica) |
| **USArrests** | One US state | 50 states | Murder, Assault, Rape (arrests per 100,000 people) and the percentage of people living in cities (UrbanPop) |
| **mtcars** | One car model | 32 cars | Mileage (mpg), horsepower (hp), weight (wt), cylinders (cyl), and gearbox type (am: 0 = automatic, 1 = manual) |

---

## 3. Part A – Grouping flowers (K-Means and Hierarchical Clustering)

**Goal:** group the 150 flowers into 3 groups using their four measurements, without using the species names.

File: [`code/02_clustering_kmeans_hierarchical_iris.R`](code/02_clustering_kmeans_hierarchical_iris.R)

```r
# 1. K-Means Clustering
data(iris)
x <- iris[,1:4]
km <- kmeans(x, centers=3)
km
plot(x[,1], x[,2], col=km$cluster, pch=19,
     xlab="Sepal Length", ylab="Sepal Width",
     main="K-Means Clustering (k=3)")

# 2. Hierarchical Clustering
hc <- hclust(dist(x))
plot(hc, main="Hierarchical Clustering Dendrogram",
     xlab="Sample Index", ylab="Distance", cex=0.6)
rect.hclust(hc, k=3)
```

**Explanation:**
- `x <- iris[,1:4]` keeps only the four number columns. The species column is left out on purpose, because clustering finds groups on its own.
- `kmeans(x, centers=3)` asks R to form 3 groups. It picks 3 centre points and keeps assigning each flower to the nearest centre.
- `plot(...)` draws the flowers as a scatter plot, with each group in a different colour.
- `dist(x)` measures how far apart every pair of flowers is. `hclust()` then joins the closest flowers step by step, forming a family-tree-like structure.
- `rect.hclust(hc, k=3)` draws boxes around the 3 main branches of that tree.

**Output – Hierarchical clustering dendrogram:**

![Dendrogram](images/03_iris_hierarchical_dendrogram.png)

**What this shows:** The tree starts with 150 individual flowers at the bottom and joins them upward. The red boxes mark the 3 groups that remain when the tree is cut into 3 branches. One group is clearly separate from the other two, and the other two are more alike.

> K-Means output: the K-Means scatter plot is produced by running the code above. It colours the 150 flowers into 3 groups.

---

## 4. Part B – Grouping US states (K-Means on USArrests)

**Goal:** group the 50 US states into 4 groups based on their crime figures and city population.

File: [`code/04_kmeans_usarrests.R`](code/04_kmeans_usarrests.R)

```r
data(USArrests)
x <- scale(USArrests)
kmodel <- kmeans(x, centers=4)
print(kmodel)
plot(x[,1], x[,2], col=kmodel$cluster, pch=19,
     xlab="Murder", ylab="Assault")
```

**Explanation:**
- `scale(USArrests)` puts all columns on the same scale. Without this, columns with large numbers (like Assault) would dominate columns with small numbers (like Murder).
- `kmeans(x, centers=4)` forms 4 groups of states.
- `print(kmodel)` shows the size of each group, the average values of each group and which state belongs to which group.
- The plot shows Murder against Assault, with each state coloured by its group.

**Output:**

![K-Means scatter](images/06_usarrests_kmeans_scatter.png)

![Cluster means and membership](images/07_usarrests_cluster_means_and_membership.png)

![Cluster summary](images/08_usarrests_cluster_summary.png)

**Group sizes found:** 13, 13, 16 and 8 states.

| Group (in my run) | Murder | Assault | UrbanPop | Rape | Size | In simple words |
|---|---|---|---|---|---|---|
| 1 | Very low | Very low | Low | Very low | 13 | Low-crime, less urban states |
| 2 | High | Very high | High | Very high | 13 | High-crime, highly urban states |
| 3 | Below average | Below average | Above average | Below average | 16 | Moderate crime, fairly urban states |
| 4 | Very high | High | Low | About average | 8 | High murder rate, less urban states |

*(The values in the output are standardized, so "very low" means well below the average of all states and "very high" means well above it. Group numbers can change each time K-Means is run.)*

**What this shows:** States with high murder tend to also have high assault, so they land in the same groups. The plot shows the groups spread along a diagonal from low-crime to high-crime states.

---

## 5. Part C – Predicting gearbox type (Decision Tree on mtcars)

**Goal:** predict whether a car has an **automatic (0)** or **manual (1)** gearbox using its mileage, horsepower, weight and number of cylinders.

Files: [`code/03_decision_tree_mtcars.R`](code/03_decision_tree_mtcars.R) and [`code/03b_decision_tree_mtcars_rpart_plot.R`](code/03b_decision_tree_mtcars_rpart_plot.R)

```r
library(rpart)
data(mtcars)

# Convert target to categorical
mtcars$am <- factor(mtcars$am)

# Build Decision Tree
dt <- rpart(am ~ mpg + hp + wt + cyl, data=mtcars)

# Print tree structure
print(dt)

# Visualize Decision Tree
plot(dt)
text(dt)
```

**Explanation:**
- `factor(mtcars$am)` tells R that gearbox type is a category (automatic or manual), not an ordinary number.
- `rpart(am ~ mpg + hp + wt + cyl, ...)` builds a decision tree. It reads as: *predict `am` using mpg, hp, wt and cyl.*
- `print(dt)` writes the tree out as text.
- `plot(dt)` and `text(dt)` draw the tree. The second file uses the `rpart.plot` package, which gives the coloured version shown below.

**Output – The decision tree:**

![Decision tree](images/04_mtcars_decision_tree.png)

**Output – The tree as text:**

![Tree printout](images/05_mtcars_tree_printout.png)

**How to read the tree:**
- It asks one question: **is the car's weight (wt) 3.18 or more?**
- **Yes (18 cars):** predicted **automatic**. Almost all of these cars (17 of 18) are automatic.
- **No (14 cars):** predicted **manual**. About 86% of these lighter cars (12 of 14) are manual.

**What this shows:** Out of mileage, horsepower, weight and cylinders, the tree found that **car weight** is the most useful clue. Heavier cars in this data are mostly automatic and lighter cars are mostly manual.

---

## 6. Part D – Predicting flower species (Naive Bayes on iris)

**Goal:** teach the computer to recognise the species of a flower from its measurements, and then test it on flowers it has not seen before.

File: [`code/05_naive_bayes_iris.R`](code/05_naive_bayes_iris.R)

```r
library(e1071)
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
```

**Explanation:**
- `set.seed(123)` makes the random split the same every time the code is run.
- The data is split into **70% for training** (105 flowers, used to teach the model) and **30% for testing** (45 flowers, kept hidden to check the model).
- `naiveBayes(Species ~ ., ...)` trains the model to predict species from all the other columns.
- `predict(model, test)` predicts the species of the 45 test flowers.
- `table(pred, test$Species)` compares predictions with the real species. `mean(pred == test$Species)` gives the share of correct predictions.

**Output:**

![Naive Bayes output](images/09_naive_bayes_confusion_matrix.png)

| Predicted \ Actual | Setosa | Versicolor | Virginica |
|---|---|---|---|
| **Setosa** | 14 | 0 | 0 |
| **Versicolor** | 0 | 18 | 0 |
| **Virginica** | 0 | 0 | 13 |

**What this shows:** All **45 test flowers** were predicted correctly (result = 1, meaning 100%). Only the diagonal numbers are filled in, which means no flower was mixed up with another species. Iris is a simple, clean dataset where the three species are quite different from each other, so a result this good is expected here and may not carry over to messier real-world data.

---

## 7. Part E – Comparing three prediction methods (iris)

**Goal:** try three different methods on the same task and compare them. The task is to decide whether a flower is **Setosa** or **not Setosa**.

File: [`code/01_classification_comparison_iris.R`](code/01_classification_comparison_iris.R)

The three methods are:
- **Logistic Regression** – draws a dividing line between the two groups.
- **KNN (k = 5)** – looks at the 5 most similar flowers and goes with the majority.
- **Decision Tree** – asks a series of yes/no questions about the measurements.

**Output:**

![Accuracy bar chart](images/01_classification_accuracy_bar_chart.png)

![Accuracy values](images/02_classification_accuracy_values.png)

| Method | Accuracy shown |
|---|---|
| Logistic Regression | 100% |
| KNN (k = 5) | 99.3% |
| Decision Tree | 96.7% |

> **Important note:** In this program, each method was tested on the **same flowers it was trained on**, without a separate test set. Because of this, these accuracy values are higher than what we would expect on new data and should be read only as a comparison of the three methods on this data. The Naive Bayes program in Part D does it the proper way, with a separate training set and test set.

**Next step:** repeat this comparison with a train/test split, as done in Part D.

---

## 8. Summary of what I learned

- **Clustering finds groups on its own.** K-Means needs the number of groups to be chosen in advance, while hierarchical clustering builds a tree that can be cut into any number of groups.
- **Standardizing data matters.** In the USArrests example, scaling the columns first stops large-number columns from dominating the grouping.
- **A decision tree is easy to explain.** The mtcars tree can be explained in one sentence: heavier cars tend to be automatic, lighter cars tend to be manual.
- **Always test on data the model has not seen.** The Naive Bayes program used a separate test set, which is a more reliable way to check a model than testing on the training data.

---

## 9. Limitations

- All datasets are **small learning datasets**, so the results show how the methods work, not conclusions about real cars, states or flowers.
- Group numbers in K-Means can change each time the code is run, because it starts from random centre points.
- The number of groups (3 for iris, 4 for USArrests) was chosen directly and not tested against other values.

---

## 10. How to run it yourself

1. Install R and RStudio.
2. Install the packages used: `install.packages(c("class", "rpart", "rpart.plot", "e1071"))`
3. Open any file from the `code/` folder in RStudio and click **Source**.

---

## 11. Project structure

```
data-grouping-and-prediction-r/
├── README.md
├── code/
│   ├── 01_classification_comparison_iris.R
│   ├── 02_clustering_kmeans_hierarchical_iris.R
│   ├── 03_decision_tree_mtcars.R
│   ├── 03b_decision_tree_mtcars_rpart_plot.R
│   ├── 04_kmeans_usarrests.R
│   └── 05_naive_bayes_iris.R
└── images/
    └── (output screenshots)
```

---

**Tools used:** R, RStudio, packages `class`, `rpart`, `rpart.plot`, `e1071`
**Author:** Swapnil Anilkumar
