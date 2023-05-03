library(readxl)
library(cluster)
library(NbClust)
library(factoextra)
library(fpc)

vehicles <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/PC Part/vehicles.xlsx") # Read the Excel file
vehicles <- vehicles[, -c(1, ncol(vehicles))]  # Remove the first and last columns
vehicles <- na.omit(vehicles) # Remove rows with missing values
vehicles <- scale(vehicles) # Scale the data
outliers <- apply(vehicles, 1, function(x) any(x > 3 | x < -3)) # Identify the outliers
vehicles <- subset(vehicles, !outliers) # Remove outliers


# <----------------------------- 1 - E ----------------------------->

# Perform PCA
pca <- prcomp(vehicles)

# Extract eigenvalues and eigenvectors
eigenvalues <- pca$sdev^2
eigenvectors <- pca$rotation

# Calculate the cumulative proportion of variance explained
cum_prop_var <- cumsum(eigenvalues / sum(eigenvalues))

# Print the number of components required to explain 92% variance
n_components <- which(cum_prop_var >= 0.92)[1]
cat("Number of components to explain 92% variance:", n_components, "\n")

# Mark the plot at 92% cumulative proportion of variance explained
plot(cum_prop_var, type = "b", xlab = "Number of PC", ylab = "Cumulative Proportion")
abline(h = 0.92, v = n_components, col = "red", lty = "dashed")

# Create a new dataset with the chosen principal components
vehicles_pca <- predict(pca, newdata = vehicles)[, 1:n_components]

vehicles_pca <- as.data.frame(vehicles_pca)


# <----------------------------- 1 - F ----------------------------->

# NBclust
NBclust <- NbClust(vehicles_pca, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")

# Perform elbow method and plot WSS for different values of k
fviz_nbclust(vehicles_pca, kmeans, method = "wss")

# Gap statistics
fviz_nbclust(vehicles_pca, kmeans, method = "gap_stat")

# Determine the optimal number of clusters using the silhouette method
fviz_nbclust(vehicles_pca, kmeans, method = "silhouette")



# <----------------------------- 1 - G ----------------------------->

# Choose the best k from the automated methods
k_pca <- 3

# Perform k-means clustering on the PCA-based dataset
km_pca <- kmeans(vehicles_pca, centers = k_pca)

# Print the cluster centers & cluster assignments for each data point
km_pca$centers
km_pca$cluster

# Create a plot of the clustering results
fviz_cluster(km_pca, data = vehicles_pca, palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point", ellipse.type = "convex", ggtheme = theme_bw(),
             main = "K-means Clustering Results"
)

# Compute the WSS, BSS and TSS
wss_pca <- sum(km_pca$withinss)
bss_pca <- sum(km_pca$betweenss)
tss_pca <- wss_pca + bss_pca

# Print the ratio of BSS over TSS
cat("WSS :", wss_pca)
cat("BSS :", bss_pca)
cat("TSS :", tss_pca)
cat("Ratio of BSS over TSS:", bss_pca / tss_pca)



# <----------------------------- 1 - H ----------------------------->

# Generate and plot the silhouette plot
sil_width_pca <- silhouette(km_pca$cluster, dist(vehicles))
plot(sil_width_pca, main = "Silhouette Plot for K-means Clustering")

# Calculate and print the average silhouette width
cat("Average Silhouette Width:", mean(sil_width_pca[,k_pca]))

# Create a colored silhouette plot
fviz_silhouette(sil_width_pca, palette = c("#2E9FDF", "#00AFBB", "#E7B800"), ggtheme = theme_bw(),
                main = "Silhouette Plot of Clustering Results")



# Calculate the Calinski-Harabasz Index
ch_index <- calinhara(vehicles_pca, km_pca$cluster)

# Print the CH index value
cat("Calinski-Harabasz Index:", ch_index)

# Visualize the clustering results
fviz_cluster(km_pca, data = vehicles_pca)
