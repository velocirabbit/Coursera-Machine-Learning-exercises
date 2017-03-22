## Machine Learning Online Class (R implementation) - Exercise 3 | Part 1: One-vs-all
require(c('rdetools', 'R.matlab'))
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = TRUE), source)  # load all fn's

input_layer_size = 400  # 20x20 input images of digits
num_labels = 10         # 10 labels, from 1 to 10 ("0" is mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
# Load training data