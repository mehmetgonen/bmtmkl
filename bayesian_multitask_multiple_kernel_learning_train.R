logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bayesian_multitask_multiple_kernel_learning_train <- function(Km, y, parameters) {
  set.seed(parameters$seed)

  T <- length(Km)
  D <- matrix(0, T, 1)
  N <- matrix(0, T, 1)
  for (t in 1:T) {
    D[t] <- dim(Km[[t]])[1]
    N[t] <- dim(Km[[t]])[2]
  }
  P <- dim(Km[[1]])[3]

  log2pi <- log(2 * pi)

  lambda <- vector("list", T)
  for (t in 1:T) {
    lambda[[t]] <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D[t], 1), beta = matrix(parameters$beta_lambda, D[t], 1))
  }
  upsilon <- list(alpha = matrix(parameters$alpha_upsilon + 0.5 * N * P, T, 1), beta = matrix(parameters$beta_upsilon, T, 1))
  a <- vector("list", T)
  for (t in 1:T) {
    a[[t]] <- list(mu = matrix(rnorm(D[t]), D[t], 1), sigma = diag(1, D[t], D[t]))
  }
  G <- vector("list", T)
  for (t in 1:T) {
    G[[t]] <- list(mu = matrix(rnorm(P * N[t]), P, N[t]), sigma = diag(1, P, P))
  }
  gamma <- list(alpha = matrix(parameters$alpha_gamma + 0.5, T, 1), beta = matrix(parameters$beta_gamma, T, 1))
  omega <- list(alpha = matrix(parameters$alpha_omega + 0.5, P, 1), beta = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(alpha = matrix(parameters$alpha_epsilon + 0.5 * N, T, 1), beta = matrix(parameters$beta_epsilon, T, 1))
  be <- list(mu = rbind(matrix(0, T, 1), matrix(1, P, 1)), sigma = diag(1, T + P, T + P))

  KmKm <- vector("list", T)
  for (t in 1:T) {
      KmKm[[t]] <- matrix(0, D[t], D[t])
      for (m in 1:P) {
          KmKm[[t]] <- KmKm[[t]] + tcrossprod(Km[[t]][,,m], Km[[t]][,,m])
      }
      Km[[t]] <- matrix(Km[[t]], D[t], N[t] * P)
  }

  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mu <- vector("list", T)
  for (t in 1:T) {
    atimesaT.mu[[t]] <- tcrossprod(a[[t]]$mu, a[[t]]$mu) + a[[t]]$sigma
  }
  GtimesGT.mu <- vector("list", T)
  for (t in 1:T) {
    GtimesGT.mu[[t]] <- tcrossprod(G[[t]]$mu, G[[t]]$mu) + N[t] * G[[t]]$sigma
  }
  btimesbT.mu <- tcrossprod(be$mu[1:T], be$mu[1:T]) + be$sigma[1:T, 1:T]
  etimeseT.mu <- tcrossprod(be$mu[(T + 1):(T + P)], be$mu[(T + 1):(T + P)]) + be$sigma[(T + 1):(T + P), (T + 1):(T + P)]
  etimesb.mu <- matrix(0, P, T)
  for (t in 1:T) {
    etimesb.mu[,t] <- be$mu[(T + 1):(T + P)] * be$mu[t] + be$sigma[(T + 1):(T + P), t]
  }
  KmtimesGT.mu <- vector("list", T)
  for (t in 1:T) {
    KmtimesGT.mu[[t]] <- Km[[t]] %*% matrix(t(G[[t]]$mu), N[t] * P, 1)
  }
  for (iter in 1:parameters$iteration) {
    # update lambda
    for (t in 1:T) {
      lambda[[t]]$beta <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mu[[t]]))
    }
    # update upsilon
    for (t in 1:T) {
      upsilon$beta[t] <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mu[[t]])) - 2 * sum(matrix(crossprod(a[[t]]$mu, Km[[t]]), N[t], P) * t(G[[t]]$mu)) + sum(diag(KmKm[[t]] %*% atimesaT.mu[[t]]))))
    }
    # update a
    for (t in 1:T) {
      a[[t]]$sigma <- chol2inv(chol(diag(as.vector(lambda[[t]]$alpha * lambda[[t]]$beta), D[t], D[t]) + upsilon$alpha[t] * upsilon$beta[t] * KmKm[[t]]))
      a[[t]]$mu <- a[[t]]$sigma %*% (upsilon$alpha[t] * upsilon$beta[t] * KmtimesGT.mu[[t]])
      atimesaT.mu[[t]] <- tcrossprod(a[[t]]$mu, a[[t]]$mu) + a[[t]]$sigma
    }
    # update G
    for (t in 1:T) {
      G[[t]]$sigma <- chol2inv(chol(diag(upsilon$alpha[t] * upsilon$beta[t], P, P) + epsilon$alpha[t] * epsilon$beta[t] * etimeseT.mu))
      G[[t]]$mu <- G[[t]]$sigma %*% (upsilon$alpha[t] * upsilon$beta[t] * t(matrix(crossprod(a[[t]]$mu, Km[[t]]), N[t], P)) + epsilon$alpha[t] * epsilon$beta[t] * (tcrossprod(be$mu[(T + 1):(T + P)], y[[t]]) - matrix(etimesb.mu[,t], P, N[t], byrow = FALSE)))
      GtimesGT.mu[[t]] <- tcrossprod(G[[t]]$mu, G[[t]]$mu) + N[t] * G[[t]]$sigma
      KmtimesGT.mu[[t]] <- Km[[t]] %*% matrix(t(G[[t]]$mu), N[t] * P, 1)
    }
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mu))
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mu))
    # update epsilon
    for (t in 1:T) {
      epsilon$beta[t] <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(y[[t]], y[[t]]) - 2 * crossprod(y[[t]], crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mu), be$mu[c(t, (T + 1):(T + P))])) + N[t] * btimesbT.mu[t, t] + sum(diag(GtimesGT.mu[[t]] %*% etimeseT.mu)) + 2 * sum(diag(crossprod(rowSums(G[[t]]$mu), etimesb.mu[,t])))))
    }
    # update b and e
    be$sigma <- rbind(cbind(diag(as.vector(gamma$alpha * gamma$beta), T, T) + diag(as.vector(N * epsilon$alpha * epsilon$beta), T, T), matrix(0, T, P)), cbind(matrix(0, P, T), diag(as.vector(omega$alpha * omega$beta), P, P)))
    for (t in 1:T) {
      be$sigma[(T + 1):(T + P), t] <- epsilon$alpha[t] * epsilon$beta[t] * rowSums(G[[t]]$mu)
      be$sigma[t, (T + 1):(T + P)] <- epsilon$alpha[t] * epsilon$beta[t] * t(rowSums(G[[t]]$mu))
      be$sigma[(T + 1):(T + P), (T + 1):(T + P)] <- be$sigma[(T + 1):(T + P), (T + 1):(T + P)] + epsilon$alpha[t] * epsilon$beta[t] * GtimesGT.mu[[t]]
    }
    be$sigma <- chol2inv(chol(be$sigma))
    be$mu <- matrix(0, T + P, 1)
    for (t in 1:T) {
      be$mu[t] <- epsilon$alpha[t] * epsilon$beta[t] * sum(y[[t]])
      be$mu[(T + 1):(T + P)] <- be$mu[(T + 1):(T + P)] + epsilon$alpha[t] * epsilon$beta[t] * G[[t]]$mu %*% y[[t]]
    }
    be$mu <- be$sigma %*% be$mu
    btimesbT.mu <- tcrossprod(be$mu[1:T], be$mu[1:T]) + be$sigma[1:T, 1:T]
    etimeseT.mu <- tcrossprod(be$mu[(T + 1):(T + P)], be$mu[(T + 1):(T + P)]) + be$sigma[(T + 1):(T + P), (T + 1):(T + P)]
    for (t in 1:T) {
        etimesb.mu[,t] <- be$mu[(T + 1):(T + P)] * be$mu[t] + be$sigma[(T + 1):(T + P), t]
    }

    if (parameters$progress == 1) {
      lb <- 0

      # p(lambda)
      for (t in 1:T) {
        lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda[[t]]$alpha) + log(lambda[[t]]$beta)) - lambda[[t]]$alpha * lambda[[t]]$beta / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      }
      # p(upsilon)
      lb <- lb + sum((parameters$alpha_upsilon - 1) * (digamma(upsilon$alpha) + log(upsilon$beta)) - upsilon$alpha * upsilon$beta / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon))
      # p(a | lambda)
      for (t in 1:T) {
        lb <- lb - 0.5 * sum(diag(diag(as.vector(lambda[[t]]$alpha * lambda[[t]]$beta), D[t], D[t]) %*% atimesaT.mu[[t]])) - 0.5 * (D[t] * log2pi - sum(digamma(lambda[[t]]$alpha) + log(lambda[[t]]$beta)))
      }
      # p(G | a, Km, upsilon)
      for (t in 1:T) {
        lb <- lb - 0.5 * sum(diag(GtimesGT.mu[[t]])) * upsilon$alpha[t] * upsilon$beta[t] + crossprod(a[[t]]$mu, KmtimesGT.mu[[t]]) * upsilon$alpha[t] * upsilon$beta[t] - 0.5 * sum(diag(KmKm[[t]] %*% atimesaT.mu[[t]])) * upsilon$alpha[t] * upsilon$beta[t] - 0.5 * N[t] * P * (log2pi - (digamma(upsilon$alpha[t]) + log(upsilon$beta[t])))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$alpha) + log(gamma$beta)) - gamma$alpha * gamma$beta / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(diag(diag(as.vector(gamma$alpha * gamma$beta), T, T) %*% btimesbT.mu)) - 0.5 * (T * log2pi - sum(digamma(gamma$alpha) + log(gamma$beta)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$alpha) + log(omega$beta)) - omega$alpha * omega$beta / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(diag(diag(as.vector(omega$alpha * omega$beta), P, P) %*% etimeseT.mu)) - 0.5 * (P * log2pi - sum(digamma(omega$alpha) + log(omega$beta)))
      # p(epsilon)
      lb <- lb + sum((parameters$alpha_epsilon - 1) * (digamma(epsilon$alpha) + log(epsilon$beta)) - epsilon$alpha * epsilon$beta / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon))
      # p(y | b, e, G, epsilon)
      for (t in 1:T) {
        lb <- lb - 0.5 * crossprod(y[[t]], y[[t]]) * epsilon$alpha[t] * epsilon$beta[t] + crossprod(y[[t]], crossprod(G[[t]]$mu, be$mu[(T + 1):(T + P)])) * epsilon$alpha[t] * epsilon$beta[t] + sum(be$mu[t] * y[[t]]) * epsilon$alpha[t] * epsilon$beta[t] - 0.5 * sum(diag(etimeseT.mu %*% GtimesGT.mu[[t]])) * epsilon$alpha[t] * epsilon$beta[t] - sum(crossprod(G[[t]]$mu, etimesb.mu[,t])) * epsilon$alpha[t] * epsilon$beta[t] - 0.5 * N[t] * btimesbT.mu[t,t] * epsilon$alpha[t] * epsilon$beta[t] - 0.5 * N[t] * (log2pi - (digamma(epsilon$alpha[t]) + log(epsilon$beta[t])))
      }

      # q(lambda)
      for (t in 1:T) {
        lb <- lb + sum(lambda[[t]]$alpha + log(lambda[[t]]$beta) + lgamma(lambda[[t]]$alpha) + (1 - lambda[[t]]$alpha) * digamma(lambda[[t]]$alpha))
      }
      # q(upsilon)
      lb <- lb + sum(upsilon$alpha + log(upsilon$beta) + lgamma(upsilon$alpha) + (1 - upsilon$alpha) * digamma(upsilon$alpha))
      # q(a)
      for (t in 1:T) {
        lb <- lb + 0.5 * (D[t] * (log2pi + 1) + logdet(a[[t]]$sigma))
      }
      # q(G)
      for (t in 1:T) {
        lb <- lb + 0.5 * N[t] * (P * (log2pi + 1) + logdet(G[[t]]$sigma))
      }
      # q(gamma)
      lb <- lb + sum(gamma$alpha + log(gamma$beta) + lgamma(gamma$alpha) + (1 - gamma$alpha) * digamma(gamma$alpha))
      # q(omega)
      lb <- lb + sum(omega$alpha + log(omega$beta) + lgamma(omega$alpha) + (1 - omega$alpha) * digamma(omega$alpha))
      # q(epsilon)
      lb <- lb + sum(epsilon$alpha + log(epsilon$beta) + lgamma(epsilon$alpha) + (1 - epsilon$alpha) * digamma(epsilon$alpha))
      # q(b, e)
      lb <- lb + 0.5 * ((T + P) * (log2pi + 1) + logdet(be$sigma))

      bounds[iter] <- lb
    }
  }
  
  if (parameters$progress == 1) {
    state <- list(lambda = lambda, upsilon = upsilon, a = a, gamma = gamma, omega = omega, epsilon = epsilon, be = be, bounds = bounds, parameters = parameters)
  }
  else {
    state <- list(lambda = lambda, upsilon = upsilon, a = a, gamma = gamma, omega = omega, epsilon = epsilon, be = be, parameters = parameters)
  }
}
