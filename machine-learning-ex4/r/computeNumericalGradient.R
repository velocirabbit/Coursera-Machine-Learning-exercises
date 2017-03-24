computeNumericalGradient = function(J, theta) {
  numgrad = matrix(0, nrow = dim(theta)[1], ncol = dim(theta)[2])
  perturb = matrix(0, nrow = dim(theta)[1], ncol = dim(theta)[2])
  e = 1e-4
  for (p in 1:length(theta1)) {
    # Set perturbation vector
    perturb[p] = e
    loss1 = J(theta - perturb)$cost
    loss2 = J(theta + perturb)$cost
    
    # Compute numerical gradient and reset perturbation vector
    numgrad[p] = (loss2 - loss1) / (2 * e)
    perturb[p] = 0
  }
  return(numgrad)
}