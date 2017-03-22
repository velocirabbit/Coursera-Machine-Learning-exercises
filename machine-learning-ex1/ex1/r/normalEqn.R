normalEqn = function(X, y) {
  Xtrans = t(X)
  theta = solve(Xtrans %*% X) %*% Xtrans %*% y
  return(theta)
}