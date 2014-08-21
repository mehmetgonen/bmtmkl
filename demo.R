source("bayesian_multitask_multiple_kernel_learning_train.R")
source("bayesian_multitask_multiple_kernel_learning_test.R")

#initalize the parameters of the algorithm
parameters <- list()

#set the hyperparameters of gamma prior used for sample weights
parameters$alpha_lambda <- 1
parameters$beta_lambda <- 1

#set the hyperparameters of gamma prior used for intermediate noise
parameters$alpha_upsilon <- 1
parameters$beta_upsilon <- 1

#set the hyperparameters of gamma prior used for bias
parameters$alpha_gamma <- 1
parameters$beta_gamma <- 1

#set the hyperparameters of gamma prior used for kernel weights
parameters$alpha_omega <- 1
parameters$beta_omega <- 1

#set the hyperparameters of gamma prior used for output noise
parameters$alpha_epsilon <- 1
parameters$beta_epsilon <- 1

### IMPORTANT ###
#For gamma priors, you can experiment with three different (alpha, beta) values
#(1, 1) => default priors
#(1e-10, 1e+10) => good for obtaining sparsity
#(1e-10, 1e-10) => good for small sample size problems (like in Nature Biotechnology paper)

#set the number of iterations
parameters$iteration <- 200

#determine whether you want to calculate and store the lower bound values
parameters$progress <- 0

#set the seed for random number generator used to initalize random variables
parameters$seed <- 1606

#set the number of tasks (e.g., the number of compounds in Nature Biotechnology paper)
T <- ??
#set the number of kernels (e.g., the number of views in Nature Biotechnology paper)
P <- ??

#initialize the kernels and outputs of each task for training
Ktrain <- vector("list", T)
ytrain <- vector("list", T)
for (t in 1:T) {
  Ktrain[[t]] <- ?? #should be an Ntra x Ntra x P matrix containing similarity values between training samples of task t
  ytrain[[t]] <- ?? #should be an Ntra x 1 matrix containing target outputs of task t
}

#perform training
state <- bayesian_multitask_multiple_kernel_learning_train(Ktrain, ytrain, parameters)

#display the kernel weights
print(state$be$mean[(T+1):(T+P)])

#initialize the kernels of each task for testing
Ktest <- vector("list", T)
for (t in 1:T) {
  Ktest[[t]] <- ?? #should be an Ntra x Ntest x P matrix containing similarity values between training and test samples of task t
}

#perform prediction
prediction <- bayesian_multitask_multiple_kernel_learning_test(Ktest, state)

#display the predictions for each task
for (t in 1:T) {
  print(prediction$y[[t]]$mean)
}
