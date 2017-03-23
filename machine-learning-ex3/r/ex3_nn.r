## Machine Learning Online Class (R implementation) - Exercise 3 | Part 2: Neural Networks
require('rdetools')
require('R.matlab')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = TRUE), source)  # load all fn's

input_layer_size = 400  # 20x20 input images of digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, from 1 to 10. "0" is mapped to label 10

## =========== Part 1: Loading and Visualizing Data =============
# Load training data
cat("Loading and visualizing data...\n")

# Get training data stored in .mat file
data = readMat('ex3data1.mat')
X = data$X
y = data$y

m = dim(X)[1]  # Number of training examples

# Randomly select 100 data points to display
rand_indices = sample(m, 100)
sel = X[rand_indices,]

displayData(sel)

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 2: Loading Pameters ================
cat("Loading saved neural network parameters...")

# Load the weights into variables Theta1 and Theta2
params = readMat('ex3weights.mat')
Theta1 = params$Theta1
Theta2 = params$Theta2

## ================= Part 3: Implement Predict =================
pred = predict(Theta1, Theta2, X)

cat("Training set accuracy:", mean(pred == y) * 100, "%\n")

readline(prompt = "Program paused. Press enter to continue.")

# To give you an idea of the network's output, you can also run through the
# examples one at a time to see what it is predicting.
rp = sample(m, m)

for (i in 1:m) {
  # Display
  cat("\nDisplaying example image...\n")
  displayData(X[rp[i],])
  
  cat("Neural network prediction: ")
  pred = predict(Theta1, Theta2, X[rp[i],])
  cat(pred, "(digit", y[rp[i]], ")\n")
  
  s = readline(prompt = "Paused - press enter to continue, q to exit.")
  if (s == '1') {
    break
  }
}
