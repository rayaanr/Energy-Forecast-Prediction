library(readxl)
library(cluster)
library(NbClust)
library(factoextra)


vehicles <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/PC Part/vehicles.xlsx")

# <----------------------------- 1 - A ----------------------------->

# Remove the "Class" & "Sample" columns & missing values
vehicles$Class <- NULL
vehicles$Samples <- NULL
vehicles <- na.omit(vehicles)

# Calculate the z-scores for each data point
z_scores <- scale(vehicles)

# Identify the data points with z-scores greater than 3 or less than -3
outliers <- which(z_scores > 3 | z_scores < -3, arr.ind = TRUE)

# Remove the identified data points from the dataset
vehicles <- vehicles[-outliers[,1],]

# Create a boxplot of the dataset
boxplot(vehicles)




# Import required libraries
library(readxl)
library(factoextra)

# Read data
vehicles <- read_excel("Dataset/vehicles.xlsx")

# Create working data set
vehicles.raw <- vehicles[2:20]

# Detect outliers by boxplot method
outliers <- boxplot(vehicles.raw[1:18], plot = TRUE)$out

# Delete outliers
vehicles.clear <- vehicles.raw[-outliers,]

# Copy class names for evaluation
classes.name <- vehicles.clear$Class

# Scale data
vehicles.scale <- scale(vehicles.clear)

# Summary and structure of scaled data
summary(vehicles.scale)
str(vehicles.scale)




NBclust <- NbClust(vehicles.scale, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")


fviz_nbclust(vehicles.scale, kmeans, method = "wss")


fviz_nbclust(vehicles.scale, kmeans, method = "gap_stat")

fviz_nbclust(vehicles.scale, kmeans, method = "silhouette")






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
k <- 2

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
# Generate the silhouette plot
sil <- silhouette(km$cluster, dist(vehicles))

# Plot the silhouette plot
plot(sil, main = "Silhouette Plot for K-means Clustering")



# Calculate the silhouette width for each observation:
sil_width <- silhouette(km$cluster, dist(vehicles))

# Create a colored silhouette plot
fviz_silhouette(sil_width, palette = c("#2E9FDF", "#00AFBB", "#E7B800"), ggtheme = theme_bw(),
                main = "Silhouette Plot of Clustering Results")

# Calculate the average silhouette width for the clustering solution:
cat("Average Silhouerre Width:", mean(sil_width[,k]))




# <----------------------------- 1 - E ----------------------------->
# https://www.youtube.com/watch?v=0Jp4gsfOLMs

# Perform PCA
pca <- prcomp(vehicles, scale = TRUE, center = TRUE)

# Extract eigenvalues and eigenvectors
eigenvalues <- pca$sdev^2
eigenvectors <- pca$rotation

# Calculate and plot the proportion of variance explained by each PC
prop_var <- eigenvalues / sum(eigenvalues)
plot(prop_var, type = "b", xlab = "Principal Component", ylab = "Proportion of Variance Explained")

# Calculate the cumulative proportion of variance explained
cum_prop_var <- cumsum(prop_var)
plot(cum_prop_var, type = "b", xlab = "Number of Principal Components", ylab = "Cumulative Proportion of Variance Explained")

# Mark the plot at 92% cumulative proportion of variance explained
abline(h = 0.92, col = "red", lty = "dashed")
abline(v = which(cum_prop_var >= 0.92)[1], col = "red", lty = "dashed")

# Print the number of components required to explain 92% variance
n_components <- which(cum_prop_var >= 0.92)[1]
cat("Number of components to explain 92% variance:", n_components, "\n")

# Create a new dataset with the chosen principal components
vehicles_pca <- as.data.frame(predict(pca, vehicles))[, 1:n_components]





# <----------------------------- 1 - F ----------------------------->

# NBclust
NBclust <- NbClust(vehicles, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")

# Elbow method
fviz_nbclust(vehicles, kmeans, method = "wss")

# Gap statistics
fviz_nbclust(vehicles, kmeans, method = "gap_stat")

# Determine the optimal number of clusters using the silhouette method
fviz_nbclust(vehicles, kmeans, method = "silhouette")



# <----------------------------- 1 - G ----------------------------->

# Choose the best k from the automated methods
k_pca <- 2

# Perform k-means clustering on the PCA-based dataset
set.seed(123)
km_pca <- kmeans(vehicles_pca, centers = k_pca)

# Show the k-means output
km_pca

# Calculate the BSS and WSS indices
bss_pca <- sum(km_pca$betweenss)
wss_pca <- sum(km_pca$withinss)

# Calculate the ratio of BSS over TSS
tss_pca <- bss_pca + wss_pca
cat("Ratio of BSS over TSS:", bss_pca / tss_pca)



# <----------------------------- 1 - H ----------------------------->

# Perform k-means clustering on the PCA dataset 
set.seed(123)
km_pca <- kmeans(vehicles_pca, centers = k_pca)

# Calculate silhouette width for each observation
sil_width_pca <- silhouette(km_pca$cluster, dist(vehicles_pca))

# Plot the silhouette plot
plot(sil_width_pca, border = NA)

# Calculate the average silhouette width for the clustering solution
mean(sil_width_pca[, k_pca])

