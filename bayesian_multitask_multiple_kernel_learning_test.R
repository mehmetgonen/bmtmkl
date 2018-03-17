bayesian_multitask_multiple_kernel_learning_test <- function(Km, state) {
  T <- length(Km)
  N <- matrix(0, T, 1)
  for (t in 1:T) {
    N[t] <- dim(Km[[t]])[2]
  }
  P <- dim(Km[[1]])[3]

  G <- vector("list", T)
  for (t in 1:T) {
    G[[t]] <- list(mu = matrix(0, P, N[t]), sigma = matrix(0, P, N[t]))
    for (m in 1:P) {
      G[[t]]$mu[m,] <- crossprod(state$a[[t]]$mu, Km[[t]][,,m])
      G[[t]]$sigma[m,] <- 1 / (state$upsilon$alpha[t] * state$upsilon$beta[t]) + diag(crossprod(Km[[t]][,,m], state$a[[t]]$sigma) %*% Km[[t]][,,m])
    } 
  }

  y <- vector("list", T)
  for (t in 1:T) {
    y[[t]] <- list(mu = matrix(0, N[t], 1), sigma = matrix(0, N[t], 1))
    y[[t]]$mu <- crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mu), state$be$mu[c(t, (T + 1):(T + P))])
    y[[t]]$sigma <- 1 / (state$epsilon$alpha[t] * state$epsilon$beta[t]) + diag(crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mu), state$be$sigma[c(t, (T + 1):(T + P)), c(t, (T + 1):(T + P))]) %*% rbind(matrix(1, 1, N[t]), G[[t]]$mu))
  }

  prediction <- list(G = G, y = y)
}
