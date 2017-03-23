randInitializeWeights = function(L_in, L_out) {
  epsilon_init = sqrt(6) / sqrt(L_in + L_out)
  W = matrix(runif(L_out * (L_in + 1)), nrow = L_out, ncol = L_in + 1)
  W = W * 2 * epsilon_init - epsilon_init
  return(W)
}