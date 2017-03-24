debugInitializeWeights = function(fan_out, fan_in) {
  W <- matrix(sin(1:(fan_out * (fan_in + 1))), nrow = fan_out, ncol = fan_in + 1)
  return(W)
}