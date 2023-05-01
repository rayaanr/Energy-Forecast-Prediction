library(readxl)
library(dplyr)
library(neuralnet)

# Import data & Rename columns
energyUsage_20 <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/MLP-NN Part/uow_consumption.xlsx") %>%
  rename("18:00" = 2, "19:00" = 3, "20:00" = 4) %>% 
  select("20:00")

# time-delayed data
t_1 = lag(energyUsage_20, 1)
t_2 = lag(energyUsage_20, 2)
t_3 = lag(energyUsage_20, 3)
t_4 = lag(energyUsage_20, 4)
t_5 = lag(energyUsage_20, 5)
t_6 = lag(energyUsage_20, 6)
t_7 = lag(energyUsage_20, 7)

# Create input/output matrices with time-delayed electricity loads
input_data_1 <- data.frame(t_1)
input_data_2 <- data.frame(t_2, t_1)
input_data_3 <- data.frame(t_4, t_3, t_2, t_1)
input_data_4 <- data.frame(t_7, t_4, t_3, t_2, t_1)
input_data_5 <- data.frame(t_7, t_6, t_5, t_4, t_3, t_2, t_1)
input_data = input_data_5 #Select the input data

output_data <- energyUsage_20

# Rename columns
#colnames(input_data) <- paste0("t_", 7:1)
colnames(output_data) <- "output"

# Normalize the data adn create the IO Matrix
io_matrix <- cbind(scale(input_data), scale(output_data))
io_matrix <- na.omit(io_matrix)

summary(io_matrix)

# Subset the data into training and testing sets
train_data <- io_matrix[1:380, ]
test_data <- io_matrix[381:nrow(io_matrix), ]

# Train MLP model with modified parameters
mlp_1 <- neuralnet(output ~ .,
                 data = train_data, hidden = 10,
                 act.fct = "logistic",
                 linear.output = FALSE)

mlp_2 <- neuralnet(output ~ .,
                 data = train_data, hidden = 20,
                 act.fct = "logistic",
                 linear.output = FALSE)

mlp_3 <- neuralnet(output ~ .,
                 data = train_data, hidden = c(10,5),
                 act.fct = "logistic",
                 linear.output = FALSE)

mlp_4 <- neuralnet(output ~ .,
                   data = train_data, hidden = c(15,20),
                   act.fct = "logistic",
                   linear.output = FALSE)

mlp_5 <- neuralnet(output ~ .,
                   data = train_data, hidden = 5,
                   act.fct = "tanh",
                   linear.output = FALSE)

mlp_6 <- neuralnet(output ~ .,
                   data = train_data, hidden = 15,
                   act.fct = "tanh",
                   linear.output = FALSE)

mlp_7 <- neuralnet(output ~ .,
                   data = train_data, hidden = c(5,10),
                   act.fct = "tanh",
                   linear.output = FALSE)

mlp_8 <- neuralnet(output ~ .,
                   data = train_data, hidden = c(10,20),
                   act.fct = "tanh",
                   linear.output = FALSE)

mlp_9 <- neuralnet(output ~ .,
                   data = train_data, hidden = 8,
                   act.fct = "relu",
                   linear.output = FALSE)

mlp_10 <- neuralnet(output ~ .,
                   data = train_data, hidden = 17,
                   act.fct = "relu",
                   linear.output = FALSE)

mlp_11 <- neuralnet(output ~ .,
                   data = train_data, hidden = c(8,17),
                   act.fct = "relu",
                   linear.output = FALSE)

mlp_12 <- neuralnet(output ~ .,
                   data = train_data, hidden = c(10,15),
                   act.fct = "relu",
                   linear.output = FALSE)


mlp_1 <- neuralnet(output ~ ., data = train_data, hidden = 10, act.fct = "logistic", linear.output = FALSE) 
mlp_2 <- neuralnet(output ~ ., data = train_data, hidden = 20, act.fct = "logistic", linear.output = FALSE) 
mlp_3 <- neuralnet(output ~ ., data = train_data, hidden = c(10,5), act.fct = "logistic", linear.output = FALSE)
mlp_4 <- neuralnet(output ~ ., data = train_data, hidden = c(15,20), act.fct = "logistic", linear.output = FALSE) 
mlp_5 <- neuralnet(output ~ ., data = train_data, hidden = 5, act.fct = "tanh", linear.output = FALSE)
mlp_6 <- neuralnet(output ~ ., data = train_data, hidden = 15, act.fct = "tanh", linear.output = FALSE)
mlp_7 <- neuralnet(output ~ ., data = train_data, hidden = c(5,10), act.fct = "tanh", linear.output = FALSE) 
mlp_8 <- neuralnet(output ~ ., data = train_data, hidden = c(10,20), act.fct = "tanh", linear.output = FALSE) 
mlp_9 <- neuralnet(output ~ ., data = train_data, hidden = 8, act.fct = "relu", linear.output = FALSE) 
mlp_10 <- neuralnet(output ~ ., data = train_data, hidden = 17, act.fct = "relu", linear.output = FALSE)
mlp_11 <- neuralnet(output ~ ., data = train_data, hidden = c(8,17), act.fct = "relu", linear.output = FALSE) 
mlp_12 <- neuralnet(output ~ ., data = train_data, hidden = c(10,15), act.fct = "relu", linear.output = FALSE)

selected_mlp = mlp_1 #Select the mlp for the rest of the prediction

# Make predictions on the test data
test_pred <- compute(selected_mlp, test_data[, -ncol(test_data)])

# Rescale the predictions back to the original scale
test_pred_rescaled <- test_pred$net.result * sd(output_data$output) + mean(output_data$output)

# Rescale the output back to the original scale
test_data_df <- as.data.frame(test_data)
test_output_rescaled <- test_data_df$output * sd(output_data$output) + mean(output_data$output)

# Create actual and predicted comparison table
comparison_table <- data.frame(Expected = test_output_rescaled, Predicted = test_pred_rescaled)


# Evaluate model performance
mse <- mean((test_pred_rescaled - test_output_rescaled)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_pred_rescaled - test_output_rescaled))
mape <- mean(abs((test_output_rescaled - test_pred_rescaled)/test_output_rescaled)) * 100
smape <- mean(2 * abs(test_pred_rescaled - test_output_rescaled) / (abs(test_pred_rescaled) + abs(test_output_rescaled))) * 100


# Print model performance
cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "% \n")
cat("sMAPE:", smape, "% \n")

plot(selected_mlp)

library(ggplot2)

# Create a data frame with predicted and observed values
plot_data <- data.frame(Predicted = test_pred_rescaled, Expected = test_output_rescaled)

# Create a scatter plot with a 45-degree line
ggplot(plot_data, aes(x = Expected, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Observed", y = "Predicted", title = "Predicted vs. Observed")

