computeCost = function(x, y, theta) {
  m = length(y)
  return(sum(((x %*% theta) - y)^2) / (2 * m))
}