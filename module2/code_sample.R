
# Load data
data <- read.csv("data.csv")

# Filter rows
filtered_data <- data[data$value > 10, ]

# Calculate mean
mean_value <- mean(filtered_data$value)

# Create a new column
filtered_data$new_col <- filtered_data$value * 2

# Loop through rows (inefficient)
for (i in 1:nrow(filtered_data)) {
  filtered_data$adjusted[i] <- filtered_data$new_col[i] + 5
}

# Plot results
plot(filtered_data$value, filtered_data$adjusted)

# Save output
write.csv(filtered_data, "output.csv")

# Print summary
print(mean_value)
``

