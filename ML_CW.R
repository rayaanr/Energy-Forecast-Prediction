# Load the data from excel file
library(readxl)
vehicles <- read_excel("/Users/rayaan/Edu/ML & DS/ML CourseWork/ML_Coursework/vehicles.xlsx")

# <----------------------------- 1 - A ----------------------------->

# Remove the "Class" column
vehicles$Class <- NULL

# To check total null values and omit them
sum(is.na(vehicles)) # Check for missing values
vehicles <- na.omit(vehicles) # Remove missing values

# Calculate z-scores for each variable
z_scores <- apply(vehicles, 2, function(x) (x - mean(x)) / sd(x))

# Identify outliers using a threshold of 3 standard deviations
outliers <- which(abs(z_scores) > 3, arr.ind = TRUE)

# Remove outliers
vehicles <- vehicles[-outliers[,1], ]

# Scale data
vehicles <- scale(vehicles)



# <----------------------------- 1 - B ----------------------------->

library(cluster)

# NBclust
library(NbClust)
set.seed(123)
NBclust <- NbClust(vehicles, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
NBclust$Best.nc # Best number of clusters

# Elbow method
library(factoextra)
fviz_nbclust(vehicles, kmeans, method = "wss") + geom_vline(xintercept = 3, linetype = 2)

# Gap statistics
set.seed(123)
gap_stat <- clusGap(vehicles, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
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
summary(sil_width)$avg.width
library(factoextra)

fviz_silhouette(sil_width, print.summary = TRUE)



# Calculate the average silhouette width for the clustering solution:
#mean(sil_width[,k])

# Visualize the silhouette plot:
#plot(sil_width)











