library(readxl)
library(cluster)
library(NbClust)
library(factoextra)

# Read the Excel file
vehicles <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/PC Part/vehicles.xlsx")

# <----------------------------- 1 - A ----------------------------->

# Remove the first and last columns
vehicles <- vehicles[, -c(1, ncol(vehicles))]

# Remove rows with missing values
vehicles <- na.omit(vehicles)

# Scale the data
vehicles <- scale(vehicles)

# Create a boxplot of the data
boxplot(vehicles, main = "Data with Outliers", ylab = "Data")

# Identify the outliers and remove them
outliers <- apply(vehicles, 1, function(x) any(x > 3 | x < -3))
vehicles <- subset(vehicles, !outliers)

summary(outliers)

boxplot(vehicles, main = "Data without Outliers", ylab = "Data")



# <----------------------------- 1 - B ----------------------------->

# NBclust
NBclust <- NbClust(vehicles, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")

# Elbow method
fviz_nbclust(vehicles, kmeans, method = "wss")

# Gap statistics
fviz_nbclust(vehicles, kmeans, method = "gap_stat")

# Determine the optimal number of clusters using the silhouette method
fviz_nbclust(vehicles, kmeans, method = "silhouette")



# <----------------------------- 1 - C ----------------------------->

# Set the number of clusters
k <- 3

# Perform k-means clustering
km <- kmeans(vehicles, centers = k)

# Print the cluster centers & cluster assignments for each data point
km$centers
km$cluster

# Create a plot of the clustering results
fviz_cluster(km, data = vehicles, 
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw(),
             main = "K-means Clustering Results"
)

# Compute the WSS, BSS and TSS
wss <- sum(km$withinss)
bss <- sum(km$betweenss)
tss <- wss + bss

# Print the ratio of BSS over TSS
cat("WSS :", wss)
cat("BSS :", bss)
cat("TSS :", tss)
cat("Ratio of BSS over TSS:", bss / tss)




# <----------------------------- 1 - D ----------------------------->

# Generate and plot the silhouette plot
sil_width <- silhouette(km$cluster, dist(vehicles))
plot(sil_width, main = "Silhouette Plot for K-means Clustering")

# Calculate and print the average silhouette width
cat("Average Silhouette Width:", mean(sil_width[,k]))

# Create a colored silhouette plot
fviz_silhouette(sil_width, palette = c("#2E9FDF", "#00AFBB", "#E7B800"), ggtheme = theme_bw(),
                main = "Silhouette Plot of Clustering Results")


