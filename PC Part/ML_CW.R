library(readxl)
vehicles <- read_excel("/Users/rayaan/Edu/ML & DS/ML CourseWork/ML_Coursework/vehicles.xlsx")

# <----------------------------- 1 - A ----------------------------->

# Remove the "Class" column
vehicles$Class <- NULL
vehicles$Samples <- NULL

# To check total null values and omit them
sum(is.na(vehicles)) # Check for missing values
vehicles <- na.omit(vehicles) # Remove missing values

# Calculate z-scores for each variable
z_scores <- apply(vehicles, 2, function(x) (x - mean(x)) / sd(x))

# Identify outliers using a threshold of 3 standard deviations
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)

# create a boxplot to visualize outliers
boxplot(vehicles, main = "Boxplot of Data with Outliers", ylab = "Data")

# Remove outliers
vehicles <- vehicles[-outliers[,1], ]

# Scale data
vehicles <- scale(vehicles)



# <----------------------------- 1 - B ----------------------------->

library(cluster)

# NBclust
library(NbClust)
set.seed(123)
NBclust <- NbClust(vehicles, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")
NBclust$Best.nc # Best number of clusters

# Elbow method
library(factoextra)
fviz_nbclust(vehicles, kmeans, method = "wss" , k.max = 6) + geom_vline(xintercept = 3, linetype = 2)

# Gap statistics
set.seed(123)
gap_stat <- clusGap(vehicles, FUN = kmeans, nstart = 25, K.max = 6, B = 50)
fviz_gap_stat(gap_stat) + geom_vline(xintercept = 3, linetype = 2)

# Silhouette method
km <- kmeans(vehicles, centers = 3, nstart = 25)
sil_h <- silhouette(km$cluster, dist(vehicles))
fviz_silhouette(sil_h) + geom_vline(xintercept = 0.25, linetype = 2)



# <----------------------------- 1 - C ----------------------------->

# Set the number of clusters
k <- 3

# Perform k-means clustering
set.seed(123)
km <- kmeans(vehicles, centers = k)

# Compute the WSS, BSS and TSS
wss <- sum(km$withinss)
bss <- sum(km$betweenss)
tss <- wss + bss

# Print the ratio of BSS over TSS
cat("Ratio of BSS over TSS:", bss / tss)




# <----------------------------- 1 - D ----------------------------->


# Calculate the silhouette width for each observation:
sil_width <- silhouette(km$cluster, dist(vehicles))
#summary(sil_width)$avg.width
#library(factoextra)
#fviz_silhouette(sil_width, print.summary = TRUE)

# Calculate the average silhouette width for the clustering solution:
mean(sil_width[,k])

# Visualize the silhouette plot:
plot(sil_width, border = "NA")



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

# We choose 5 principal components because they explain more than 92% of the variance in the data



# <----------------------------- 1 - F ----------------------------->

#vehicles_pca <- scale(vehicles_pca)

# NBclust
set.seed(123)
NBclust_pca <- NbClust(vehicles_pca, distance = "euclidean", min.nc = 2, max.nc = 6, method = "kmeans")
NBclust_pca$Best.nc # Best number of clusters

# Elbow method
fviz_nbclust(vehicles_pca, kmeans, method = "wss" , k.max = 6) + geom_vline(xintercept = 3, linetype = 2)

# Gap statistics
set.seed(123)
gap_stat_pca <- clusGap(vehicles_pca, FUN = kmeans, nstart = 25, K.max = 6, B = 50)
fviz_gap_stat(gap_stat_pca) + geom_vline(xintercept = 3, linetype = 2)

# Silhouette method
km_pca <- kmeans(vehicles_pca, centers = 3, nstart = 25)
sil_h_pca <- silhouette(km_pca$cluster, dist(vehicles_pca))
fviz_silhouette(sil_h_pca) + geom_vline(xintercept = 0.25, linetype = 2)



# <----------------------------- 1 - G ----------------------------->

# Choose the best k from the automated methods
k_pca <- 3

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










