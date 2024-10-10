# Load necessary libraries
library(glmnet)
library(readr)
library(ggplot2)
library(reshape2)  # for data reshaping



# Load the data from a CSV file
data <- read_csv("~/Project/StatsLab/regression/lasso/Simulated_Dataset.csv")

# Prepare the matrix of predictors and the response variable
x <- as.matrix(data[, c('x1', 'x2', 'x3')])
y <- data$y

# Define a sequence of lambda values
lambda_values <- seq(0, 25, length.out = 1000)

# Fit the linear model over a range of lambda values with intercept = FALSE
fit <- glmnet(x, y, alpha = 1, lambda = lambda_values, intercept = FALSE)  # alpha=1 for lasso

# Extract coefficients at each lambda (excluding intercept)
coefficients_df <- sapply(1:length(fit$lambda), function(i) as.numeric(coef(fit, s = fit$lambda[i])))

# Transpose and convert to a data frame
coefficients_df <- as.data.frame(t(coefficients_df))

# Remove the intercept column and rename remaining columns
coefficients_df <- coefficients_df[, -1]  # Remove the intercept column
colnames(coefficients_df) <- c("x1", "x2", "x3")

# Add the Lambda column
coefficients_df$Lambda <- fit$lambda

# Reshape for plotting with ggplot2
coefficients_long <- melt(coefficients_df, id.vars = "Lambda", variable.name = "Coefficient", value.name = "Value")

# Plot the coefficients (without intercept)
ggplot(coefficients_long, aes(x = Lambda, y = Value, color = Coefficient)) + 
    geom_line() + 
    labs(title = "Coefficient Paths (Without Intercept)", x = "Lambda", y = "Coefficient Value") + 
    theme_minimal()
