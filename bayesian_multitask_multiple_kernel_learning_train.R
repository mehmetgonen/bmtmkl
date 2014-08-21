# Mehmet Gonen (mehmet.gonen@gmail.com)

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
    lambda[[t]] <- list(shape = matrix(parameters$alpha_lambda + 0.5, D[t], 1), scale = matrix(parameters$beta_lambda, D[t], 1))
  }
  upsilon <- list(shape = matrix(parameters$alpha_upsilon + 0.5 * N * P, T, 1), scale = matrix(parameters$beta_upsilon, T, 1))
  a <- vector("list", T)
  for (t in 1:T) {
    a[[t]] <- list(mean = matrix(rnorm(D[t]), D[t], 1), covariance = diag(1, D[t], D[t]))
  }
  G <- vector("list", T)
  for (t in 1:T) {
    G[[t]] <- list(mean = matrix(rnorm(P * N[t]), P, N[t]), covariance = diag(1, P, P))
  }
  gamma <- list(shape = matrix(parameters$alpha_gamma + 0.5, T, 1), scale = matrix(parameters$beta_gamma, T, 1))
  omega <- list(shape = matrix(parameters$alpha_omega + 0.5, P, 1), scale = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(shape = matrix(parameters$alpha_epsilon + 0.5 * N, T, 1), scale = matrix(parameters$beta_epsilon, T, 1))
  be <- list(mean = rbind(matrix(0, T, 1), matrix(1, P, 1)), covariance = diag(1, T + P, T + P))

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

  atimesaT.mean <- vector("list", T)
  for (t in 1:T) {
    atimesaT.mean[[t]] <- tcrossprod(a[[t]]$mean, a[[t]]$mean) + a[[t]]$covariance
  }
  GtimesGT.mean <- vector("list", T)
  for (t in 1:T) {
    GtimesGT.mean[[t]] <- tcrossprod(G[[t]]$mean, G[[t]]$mean) + N[t] * G[[t]]$covariance
  }
  btimesbT.mean <- tcrossprod(be$mean[1:T], be$mean[1:T]) + be$covariance[1:T, 1:T]
  etimeseT.mean <- tcrossprod(be$mean[(T + 1):(T + P)], be$mean[(T + 1):(T + P)]) + be$covariance[(T + 1):(T + P), (T + 1):(T + P)]
  etimesb.mean <- matrix(0, P, T)
  for (t in 1:T) {
    etimesb.mean[,t] <- be$mean[(T + 1):(T + P)] * be$mean[t] + be$covariance[(T + 1):(T + P), t]
  }
  KmtimesGT.mean <- vector("list", T)
  for (t in 1:T) {
    KmtimesGT.mean[[t]] <- Km[[t]] %*% matrix(t(G[[t]]$mean), N[t] * P, 1)
  }
  for (iter in 1:parameters$iteration) {
    # update lambda
    for (t in 1:T) {
      lambda[[t]]$scale <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mean[[t]]))
    }
    # update upsilon
    for (t in 1:T) {
      upsilon$scale[t] <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mean[[t]])) - 2 * sum(matrix(crossprod(a[[t]]$mean, Km[[t]]), N[t], P) * t(G[[t]]$mean)) + sum(diag(KmKm[[t]] %*% atimesaT.mean[[t]]))))
    }
    # update a
    for (t in 1:T) {
      a[[t]]$covariance <- chol2inv(chol(diag(as.vector(lambda[[t]]$shape * lambda[[t]]$scale), D[t], D[t]) + upsilon$shape[t] * upsilon$scale[t] * KmKm[[t]]))
      a[[t]]$mean <- a[[t]]$covariance %*% (upsilon$shape[t] * upsilon$scale[t] * KmtimesGT.mean[[t]])
      atimesaT.mean[[t]] <- tcrossprod(a[[t]]$mean, a[[t]]$mean) + a[[t]]$covariance
    }
    # update G
    for (t in 1:T) {
      G[[t]]$covariance <- chol2inv(chol(diag(upsilon$shape[t] * upsilon$scale[t], P, P) + epsilon$shape[t] * epsilon$scale[t] * etimeseT.mean))
      G[[t]]$mean <- G[[t]]$covariance %*% (upsilon$shape[t] * upsilon$scale[t] * t(matrix(crossprod(a[[t]]$mean, Km[[t]]), N[t], P)) + epsilon$shape[t] * epsilon$scale[t] * (tcrossprod(be$mean[(T + 1):(T + P)], y[[t]]) - matrix(etimesb.mean[,t], P, N[t], byrow = FALSE)))
      GtimesGT.mean[[t]] <- tcrossprod(G[[t]]$mean, G[[t]]$mean) + N[t] * G[[t]]$covariance
      KmtimesGT.mean[[t]] <- Km[[t]] %*% matrix(t(G[[t]]$mean), N[t] * P, 1)
    }
    # update gamma
    gamma$scale <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mean))
    # update omega
    omega$scale <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mean))
    # update epsilon
    for (t in 1:T) {
      epsilon$scale[t] <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(y[[t]], y[[t]]) - 2 * crossprod(y[[t]], crossprod(rbind(matrix(1, 1, N[t]), G[[t]]$mean), be$mean[c(t, (T + 1):(T + P))])) + N[t] * btimesbT.mean[t, t] + sum(diag(GtimesGT.mean[[t]] %*% etimeseT.mean)) + 2 * sum(diag(crossprod(rowSums(G[[t]]$mean), etimesb.mean[,t])))))
    }
    # update b and e
    be$covariance <- rbind(cbind(diag(as.vector(gamma$shape * gamma$scale), T, T) + diag(as.vector(N * epsilon$shape * epsilon$scale), T, T), matrix(0, T, P)), cbind(matrix(0, P, T), diag(as.vector(omega$shape * omega$scale), P, P)))
    for (t in 1:T) {
      be$covariance[(T + 1):(T + P), t] <- epsilon$shape[t] * epsilon$scale[t] * rowSums(G[[t]]$mean)
      be$covariance[t, (T + 1):(T + P)] <- epsilon$shape[t] * epsilon$scale[t] * t(rowSums(G[[t]]$mean))
      be$covariance[(T + 1):(T + P), (T + 1):(T + P)] <- be$covariance[(T + 1):(T + P), (T + 1):(T + P)] + epsilon$shape[t] * epsilon$scale[t] * GtimesGT.mean[[t]]
    }
    be$covariance <- chol2inv(chol(be$covariance))
    be$mean <- matrix(0, T + P, 1)
    for (t in 1:T) {
      be$mean[t] <- epsilon$shape[t] * epsilon$scale[t] * sum(y[[t]])
      be$mean[(T + 1):(T + P)] <- be$mean[(T + 1):(T + P)] + epsilon$shape[t] * epsilon$scale[t] * G[[t]]$mean %*% y[[t]]
    }
    be$mean <- be$covariance %*% be$mean
    btimesbT.mean <- tcrossprod(be$mean[1:T], be$mean[1:T]) + be$covariance[1:T, 1:T]
    etimeseT.mean <- tcrossprod(be$mean[(T + 1):(T + P)], be$mean[(T + 1):(T + P)]) + be$covariance[(T + 1):(T + P), (T + 1):(T + P)]
    for (t in 1:T) {
        etimesb.mean[,t] <- be$mean[(T + 1):(T + P)] * be$mean[t] + be$covariance[(T + 1):(T + P), t]
    }

    if (parameters$progress == 1) {
      lb <- 0

      # p(lambda)
      for (t in 1:T) {
        lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda[[t]]$shape) + log(lambda[[t]]$scale)) - lambda[[t]]$shape * lambda[[t]]$scale / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      }
      # p(upsilon)
      lb <- lb + sum((parameters$alpha_upsilon - 1) * (digamma(upsilon$shape) + log(upsilon$scale)) - upsilon$shape * upsilon$scale / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon))
      # p(a | lambda)
      for (t in 1:T) {
        lb <- lb - 0.5 * sum(diag(diag(as.vector(lambda[[t]]$shape * lambda[[t]]$scale), D[t], D[t]) %*% atimesaT.mean[[t]])) - 0.5 * (D[t] * log2pi - sum(log(lambda[[t]]$shape * lambda[[t]]$scale)))
      }
      # p(G | a, Km, upsilon)
      for (t in 1:T) {
        lb <- lb - 0.5 * sum(diag(GtimesGT.mean[[t]])) * upsilon$shape[t] * upsilon$scale[t] + crossprod(a[[t]]$mean, KmtimesGT.mean[[t]]) * upsilon$shape[t] * upsilon$scale[t] - 0.5 * sum(diag(KmKm[[t]] %*% atimesaT.mean[[t]])) * upsilon$shape[t] * upsilon$scale[t] - 0.5 * N[t] * P * (log2pi - log(upsilon$shape[t] * upsilon$scale[t]))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$shape) + log(gamma$scale)) - gamma$shape * gamma$scale / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(diag(diag(as.vector(gamma$shape * gamma$scale), T, T) %*% btimesbT.mean)) - 0.5 * (T * log2pi - sum(log(gamma$shape * gamma$scale)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$shape) + log(omega$scale)) - omega$shape * omega$scale / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(diag(diag(as.vector(omega$shape * omega$scale), P, P) %*% etimeseT.mean)) - 0.5 * (P * log2pi - sum(log(omega$shape * omega$scale)))
      # p(epsilon)
      lb <- lb + sum((parameters$alpha_epsilon - 1) * (digamma(epsilon$shape) + log(epsilon$scale)) - epsilon$shape * epsilon$scale / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon))
      # p(y | b, e, G, epsilon)
      for (t in 1:T) {
        lb <- lb - 0.5 * crossprod(y[[t]], y[[t]]) * epsilon$shape[t] * epsilon$scale[t] + crossprod(y[[t]], crossprod(G[[t]]$mean, be$mean[(T + 1):(T + P)])) * epsilon$shape[t] * epsilon$scale[t] + sum(be$mean[t] * y[[t]]) * epsilon$shape[t] * epsilon$scale[t] - 0.5 * sum(diag(etimeseT.mean %*% GtimesGT.mean[[t]])) * epsilon$shape[t] * epsilon$scale[t] - sum(crossprod(G[[t]]$mean, etimesb.mean[,t])) * epsilon$shape[t] * epsilon$scale[t] - 0.5 * N[t] * btimesbT.mean[t,t] * epsilon$shape[t] * epsilon$scale[t] - 0.5 * N[t] * (log2pi - log(epsilon$shape[t] * epsilon$scale[t]))
      }

      # q(lambda)
      for (t in 1:T) {
        lb <- lb + sum(lambda[[t]]$shape + log(lambda[[t]]$scale) + lgamma(lambda[[t]]$shape) + (1 - lambda[[t]]$shape) * digamma(lambda[[t]]$shape))
      }
      # q(upsilon)
      lb <- lb + sum(upsilon$shape + log(upsilon$scale) + lgamma(upsilon$shape) + (1 - upsilon$shape) * digamma(upsilon$shape))
      # q(a)
      for (t in 1:T) {
        lb <- lb + 0.5 * (D[t] * (log2pi + 1) + logdet(a[[t]]$covariance))
      }
      # q(G)
      for (t in 1:T) {
        lb <- lb + 0.5 * N[t] * (P * (log2pi + 1) + logdet(G[[t]]$covariance))
      }
      # q(gamma)
      lb <- lb + sum(gamma$shape + log(gamma$scale) + lgamma(gamma$shape) + (1 - gamma$shape) * digamma(gamma$shape))
      # q(omega)
      lb <- lb + sum(omega$shape + log(omega$scale) + lgamma(omega$shape) + (1 - omega$shape) * digamma(omega$shape))
      # q(epsilon)
      lb <- lb + sum(epsilon$shape + log(epsilon$scale) + lgamma(epsilon$shape) + (1 - epsilon$shape) * digamma(epsilon$shape))
      # q(b, e)
      lb <- lb + 0.5 * ((T + P) * (log2pi + 1) + logdet(be$covariance))

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
