library(readxl)
library(dplyr)
library(neuralnet)
library(ggplot2)

# Import data & Rename columns
energyUsage <- read_excel("/Users/rayaan/Edu/ML & DS/ML Coursework/MLP-NN Part/uow_consumption.xlsx") %>%
  rename("18:00" = 2, "19:00" = 3, "20:00" = 4)  %>% 
  select("18:00", "19:00", "20:00")

# time-delayed data
t1_18 = lag(energyUsage$`18:00`, 1)
t2_18 = lag(energyUsage$`18:00`, 2)
t3_18 = lag(energyUsage$`18:00`, 3)
t4_18 = lag(energyUsage$`18:00`, 4)
t7_18 = lag(energyUsage$`18:00`, 7)
t1_19 = lag(energyUsage$`19:00`, 1)
t2_19 = lag(energyUsage$`19:00`, 2)
t3_19 = lag(energyUsage$`19:00`, 3)
t4_19 = lag(energyUsage$`19:00`, 4)
t7_19 = lag(energyUsage$`19:00`, 7)
t1_20 = lag(energyUsage$`20:00`, 1)
t2_20 = lag(energyUsage$`20:00`, 2)
t3_20 = lag(energyUsage$`20:00`, 3)
t4_20 = lag(energyUsage$`20:00`, 4)
t7_20 = lag(energyUsage$`20:00`, 7)

# Create input/output matrices with time-delayed electricity loads

input_data_1 <- data.frame(t1_18, t2_18, t3_18, t4_18, t7_18, t1_19, t2_19, t3_19,
                         t4_19, t7_19, t1_20, t2_20, t3_20, t4_20, t7_20)
input_data_2 <- data.frame(t1_18, t1_19, t1_20)
input_data_3 <- data.frame(t1_18, t2_18, t3_18, t4_18, t1_19, t2_19, t3_19, t4_19)

input_data = input_data_1 #Select the input data

output_data <- energyUsage["20:00"]
colnames(output_data) <- "output"

# Normalize the data
io_matrix <- cbind(scale(input_data_1), scale(output_data))
io_matrix_1 <- cbind(scale(input_data_2), scale(output_data))
io_matrix_2 <- cbind(scale(input_data_3), scale(output_data))

io_matrix <- na.omit(io_matrix)

# Subset the data into training and testing sets
train_data <- io_matrix[1:380, ]
test_data <- io_matrix[381:nrow(io_matrix), ]

# Train MLP model with modified parameters
mlp_1 <- neuralnet(output ~  t1_18 + t2_18 + t3_18 + t4_18 + t7_18 + t1_19 + t2_19 + 
                     t3_19 + t4_19 + t7_19 + t1_20 + t2_20 + t3_20 + t4_20 + t7_20,
                 data = train_data, hidden = 3,
                 act.fct = "logistic",
                 linear.output = FALSE,
                 stepmax = 1e5)

mlp_2 <- neuralnet(output ~ t1_18 + t2_18 + t3_18 + t4_18 + t1_19 + t2_19 + t3_19 + t4_19,
                 data = train_data, hidden = c(3,2), act.fct = "logistic",
                 linear.output = FALSE, stepmax = 1e5)

mlp_3 <- neuralnet(output ~ t1_18 + t2_18 + t3_18 + t4_18 + t1_19 + t2_19 + t3_19 + t4_19,
                 data = train_data, hidden = 5, act.fct = "tanh",
                 linear.output = FALSE, stepmax = 1e5)

mlp_4 <- neuralnet(output ~ t1_18 + t2_18 + t3_18 + t4_18 + t7_18 + t1_19 + t2_19 + 
                     t3_19 + t4_19 + t7_19 + t1_20 + t2_20 + t3_20 + t4_20 + t7_20,
                 data = train_data, hidden = c(5,7), act.fct = "logistic",
                 linear.output = FALSE, stepmax = 1e5)

mlp_5 <- neuralnet(output ~ t1_18 + t1_19 + t1_20,
                 data = train_data, hidden = 3, act.fct = "tanh",
                 linear.output = FALSE, stepmax = 1e5)


selected_mlp = mlp_1 #Select the mlp for the rest of the prediction

# Make predictions on the test data & Rescale the predictions back to the original scale
test_pred <- neuralnet::compute(selected_mlp, test_data[, -ncol(test_data)])

# Rescale the predictions back to the original scale
test_pred_rescaled <- test_pred$net.result * sd(output_data$output) + mean(output_data$output)

# Rescale the output back to the original scale
test_data_df <- as.data.frame(test_data)
test_output_rescaled <- test_data_df$output * sd(output_data$output) + mean(output_data$output)

# Create actual and predicted comparison table
comparison_table <- data.frame(Actual = test_output_rescaled, Predicted = test_pred_rescaled)

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



# Create a data frame with predicted and observed values
plot_data <- data.frame(Predicted = test_pred_rescaled, Observed = test_output_rescaled)

# Create a scatter plot with a 45-degree line
ggplot(plot_data, aes(x = Observed, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual", y = "Predicted", title = "Predicted vs. Actual (NARX)")



#input_data <- data.frame(t4_18 = lag(energyUsage$`18:00`, 4),
#                        t4_19 = lag(energyUsage$`19:00`, 4),
#                        t4_20 = lag(energyUsage$`20:00`, 4),
#                        t3_18 = lag(energyUsage$`18:00`, 3),
#                        t3_19 = lag(energyUsage$`19:00`, 3),
#                        t3_20 = lag(energyUsage$`20:00`, 3),
#                        t2_18 = lag(energyUsage$`18:00`, 2),
#                        t2_19 = lag(energyUsage$`19:00`, 2),
#                        t2_20 = lag(energyUsage$`20:00`, 2),
#                        t1_18 = lag(energyUsage$`18:00`, 1),
#                        t1_19 = lag(energyUsage$`19:00`, 1),
#                        t1_20 = lag(energyUsage$`20:00`, 1)
#)
