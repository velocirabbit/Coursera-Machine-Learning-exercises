sigmoidGradient = function(z) {
  sig_func <- sigmoid(z)
  return(sig_func * (1 - sig_func))
}