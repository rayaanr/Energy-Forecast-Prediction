library(readxl)
library(dplyr)
library(neuralnet)

# Import data & Rename columns
data <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/MLP-NN Part/uow_consumption.xlsx") %>%
  rename("18:00" = 2, "19:00" = 3, "20:00" = 4) %>%
  select("20:00")

# Create input/output matrices with time-delayed electricity loads
# input_data <- data.frame(sapply(1:7, function(x) lag(data, x)))
input_data <- data.frame(t_7 = lag(data, 7),
                         t_6 = lag(data, 6),
                         t_5 = lag(data, 5),
                         t_4 = lag(data, 4),
                         t_3 = lag(data, 3),
                         t_2 = lag(data, 2),
                         t_1 = lag(data, 1))
output_data <- data

# Rename columns
colnames(input_data) <- paste0("t_", 1:7)
colnames(output_data) <- "output"

# Normalize the data
io_matrix <- cbind(scale(input_data), scale(output_data))
io_matrix <- na.omit(io_matrix)

# Subset the data into training and testing sets
train_data <- io_matrix[1:380, ]
test_data <- io_matrix[381:nrow(io_matrix), ]

# Train MLP model with modified parameters
mlp <- neuralnet(output ~ ., data = train_data, hidden = c(10,5), linear.output = FALSE)

# Make predictions on the test data & Rescale the predictions back to the original scale
test_pred <- compute(mlp, test_data[, 1:7])
test_pred_rescale <- as.data.frame(scale(test_pred$net.result,
                                          center = attr(scale(output_data), "scaled:center"),
                                          scale = attr(scale(output_data), "scaled:scale")))

# Evaluate model performance
mse <- mean((test_data[, "output"] - test_pred_rescale[, 1])^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data[, "output"] - test_pred_rescale[, 1]))
mape <- mean(abs((test_data[, "output"] - test_pred_rescale[, 1])/test_data[, "output"]))*100
smape <- mean(200*abs(test_data[, "output"] - test_pred_rescale[, 1])/(abs(test_data[, "output"]) + abs(test_pred_rescale[, 1])))

cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "\n")
cat("sMAPE:", smape, "\n")

plot(mlp)
