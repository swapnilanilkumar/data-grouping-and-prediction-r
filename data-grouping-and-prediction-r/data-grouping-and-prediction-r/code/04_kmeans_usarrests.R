data(USArrests)
x <- scale(USArrests)
kmodel <- kmeans(x, centers=4)
print(kmodel)
plot(x[,1], x[,2], col=kmodel$cluster, pch=19,
     xlab="Murder", ylab="Assault")


