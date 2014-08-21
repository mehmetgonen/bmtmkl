# Mehmet Gonen (mehmet.gonen@gmail.com)

bayesian_multitask_multiple_kernel_learning_test <- function(Km, state) {
  T <- length(Km)
  N <- matrix(0, T, 1)
  for (t in 1:T) {
    N[t] <- dim(Km[[t]])[2]
  }
  P <- dim(Km[[1]])[3]

  G <- vector("list", T)
  for (t in 1:T) {
    G[[t]] <- list(mean = matrix(0, P, N[t]), covariance = matrix(0, P, N[t]))
    for (m in 1:P) {
      G[[t]]$mean[m,] <- crossprod(state$a[[t]]$mean, Km[[t]][,,m])
      G[[t]]$covariance[m,] <- 1 / (state$upsilon$shape[t] * state$upsilon$scale[t]) + diag(crossprod(Km[[t]][,,m], state$a[[t]]$covariance) %*% Km[[t]][,,m])
    } 
  }

  y <- vector("list", T)
  for (t in 1:T) {
    y[[t]] <- list(mean = matrix(0, N[t], 1), covariance = matrix(0, N[t], 1))
    y[[t]]$mean <- crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mean), state$be$mean[c(t, (T + 1):(T + P))])
    y[[t]]$covariance <- 1 / (state$epsilon$shape[t] * state$epsilon$scale[t]) + diag(crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mean), state$be$covariance[c(t, (T + 1):(T + P)), c(t, (T + 1):(T + P))]) %*% rbind(matrix(1, 1, N[t]), G[[t]]$mean))
  }

  prediction <- list(y = y)
}
