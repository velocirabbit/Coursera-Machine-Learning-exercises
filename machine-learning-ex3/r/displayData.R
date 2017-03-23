displayData = function(X, example_width = 0) {
  if (example_width == 0) {
    example_width = round(sqrt(dim(X)[2]))
  }
  
  # R colormaps?
  
  # Compute rows, cols
  m = dim(X)[1]; n = dim(X)[2]
  example_height = (n / example_width)
  
  # Compute number of items to display
  display_rows = floor(sqrt(m))
  display_cols = ceiling(m / display_rows)
  
  # Between images padding
  pad = 1
  
  # Setup blank display
  display_array = -matrix(1, pad + display_rows * (example_height + pad),
                             pad + display_cols * (example_width + pad))
  
  # Copy each example into a patch on the display array
  curr_ex = 1
  for (j in 1:display_rows) {
    for (i in 1:display_cols) {
      if (curr_ex > m) {
        break
      }
      
      # Copy the patch
      # Get the max value of the patch
      max_val = max(abs(X[curr_ex,]))
      hts = pad + (j - 1) * (example_height + pad) + (1:example_height)
      wts = pad + (i - 1) * (example_width + pad) + (1:example_width)
      
      rawimg = matrix(X[curr_ex,], nrow = example_width, ncol = example_height)
      imgex = t(apply(rawimg, 2, rev)) / max_val  # Rotates image 90 degrees clockwise
      
      display_array[hts, wts] = imgex
      curr_ex = curr_ex + 1
    }
    if (curr_ex > m) {
      break
    }
  }
  # Display image
  ximg = dim(display_array)[1]
  yimg = dim(display_array)[2]
  image(1:ximg, 1:yimg, display_array, col = gray.colors(12))
}