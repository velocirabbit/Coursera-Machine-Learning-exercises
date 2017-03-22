plotData = function(X, y, xlabel, ylabel) {
  pos = y == 1
  neg = y == 0
  
  plot(X[pos, 1], X[pos, 2], col = 'black', pch = 3, xlab = xlabel, ylab = ylabel)
  points(X[neg, 1], X[neg, 2], col = 'orange', pch = 20)
}