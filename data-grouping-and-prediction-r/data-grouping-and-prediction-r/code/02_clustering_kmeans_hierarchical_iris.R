#1. K-Means Clustering
data(iris)
x <- iris[,1:4]
km <- kmeans(x, centers=3)
km
plot(x[,1], x[,2], col=km$cluster, pch=19,
     xlab="Sepal Length",
     ylab="Sepal Width",
     main="K-Means Clustering (k=3)")

#2. Hierarchical Clustering
data(iris)
x <- iris[,1:4]
hc <- hclust(dist(x))
plot(hc,
     main="Hierarchical Clustering Dendrogram",
     xlab="Sample Index",
     ylab="Distance",
     cex=0.6)
rect.hclust(hc, k=3)