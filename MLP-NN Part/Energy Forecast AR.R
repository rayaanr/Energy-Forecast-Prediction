library(readxl)
library(dplyr)
library(neuralnet)
library(ggplot2)

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
input_data <- data.frame(t_7, t_6, t_5, t_4, t_3, t_2, t_1)
input_data_1 <- data.frame(t_3, t_2, t_1)
input_data_2 <- data.frame(t_4, t_3, t_2, t_1)
input_data_3 <- data.frame(t_7, t_4, t_3, t_2, t_1)

colnames(input_data) <- paste0("t_", 7:1)
output_data <- energyUsage_20
colnames(output_data) <- "output"

# Normalize the data adn create the IO Matrix (Run only required matrix)
io_matrix <- cbind(scale(input_data), scale(output_data))
io_matrix_1 <- cbind(scale(input_data_1), scale(output_data))
io_matrix_2 <- cbind(scale(input_data_2), scale(output_data))
io_matrix_3 <- cbind(scale(input_data_3), scale(output_data))

io_matrix <- na.omit(io_matrix)
summary(io_matrix)

# Subset the data into training and testing sets
train_data <- io_matrix[1:380, ]
test_data <- io_matrix[381:nrow(io_matrix), ]

# Train MLP model with modified parameters
mlp_1 <- neuralnet(output ~ t_3 + t_2 + t_1,
                 data = train_data, hidden = 3,
                 act.fct = "logistic",
                 linear.output = FALSE,
                 stepmax = 1e7)

mlp_2 <- neuralnet(output ~ t_3 + t_2 + t_1, 
                   data = train_data, hidden = 5, act.fct = "logistic", linear.output = FALSE, stepmax = 1e7) 

mlp_3 <- neuralnet(output ~ t_3 + t_2 + t_1, 
                   data = train_data, hidden = c(3,2), act.fct = "logistic", linear.output = FALSE, stepmax = 1e7)

mlp_4 <- neuralnet(output ~ t_3 + t_2 + t_1,
                   data = train_data, hidden = c(5,7), act.fct = "logistic", linear.output = FALSE, stepmax = 1e7) 

mlp_5 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                   data = train_data, hidden = 3, act.fct = "logistic", linear.output = FALSE, stepmax = 1e7)

mlp_6 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                   data = train_data, hidden = 5, act.fct = "logistic", linear.output = FALSE, stepmax = 1e7)

mlp_7 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                   data = train_data, hidden = c(3,2), act.fct = "logistic", linear.output = FALSE, stepmax = 1e7) 

mlp_8 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                   data = train_data, hidden = c(5,7), act.fct = "logistic", linear.output = FALSE, stepmax = 1e7) 

mlp_9 <- neuralnet(output ~ t_3 + t_2 + t_1,
                   data = train_data, hidden = 3, act.fct = "tanh", linear.output = FALSE, stepmax = 1e7) 

mlp_10 <- neuralnet(output ~ t_3 + t_2 + t_1,
                    data = train_data, hidden = 5, act.fct = "tanh", linear.output = FALSE, stepmax = 1e7)

mlp_11 <- neuralnet(output ~ t_3 + t_2 + t_1,
                    data = train_data, hidden = c(3,2), act.fct = "tanh", linear.output = FALSE, stepmax = 1e7) 

mlp_12 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                    data = train_data, hidden = 3, act.fct = "tanh", linear.output = FALSE, stepmax = 1e7)

mlp_13 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                    data = train_data, hidden = 5, act.fct = "tanh", linear.output = FALSE, stepmax = 1e7)

mlp_14 <- neuralnet(output ~ t_7 + t_6 + t_5 + t_4 + t_3 + t_2 + t_1,
                    data = train_data, hidden = c(3,2), act.fct = "tanh", linear.output = FALSE, stepmax = 1e7)

#Select the mlp to do the rest of the process
mlp <- mlp_5

# Make predictions on the test data
test_pred <- neuralnet::compute(mlp, test_data[, -ncol(test_data)])

# Backscale the predictions back to the original scale
test_pred_rescaled <- test_pred$net.result * sd(output_data$output) + mean(output_data$output)

# Backscale the output back to the original scale
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

plot(mlp)


# Create a data frame with predicted and observed values
plot_data <- data.frame(Predicted = test_pred_rescaled, Expected = test_output_rescaled)

# Create a scatter plot with a 45-degree line
ggplot(plot_data, aes(x = Expected, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual", y = "Predicted", title = "Predicted vs. Actual (AR)")

