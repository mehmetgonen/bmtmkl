% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bayesian_multitask_multiple_kernel_learning_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    T = length(Km);
    D = zeros(T, 1);
    N = zeros(T, 1);
    for t = 1:T
        D(t) = size(Km{t}, 1);
        N(t) = size(Km{t}, 2);
    end
    P = size(Km{1}, 3);
    
    log2pi = log(2 * pi);

    lambda = cell(1, T);
    for t = 1:T
        lambda{t}.shape = (parameters.alpha_lambda + 0.5) * ones(D(t), 1);
        lambda{t}.scale = parameters.beta_lambda * ones(D(t), 1);
    end
    upsilon.shape = (parameters.alpha_upsilon + 0.5 * N * P) .* ones(T, 1);
    upsilon.scale = parameters.beta_upsilon * ones(T, 1);
    a = cell(1, T);    
    for t = 1:T
        a{t}.mean = randn(D(t), 1);
        a{t}.covariance = eye(D(t), D(t));
    end
    G = cell(1, T);
    for t = 1:T
        G{t}.mean = randn(P, N(t));
        G{t}.covariance = eye(P, P);
    end
    gamma.shape = (parameters.alpha_gamma + 0.5) * ones(T, 1);
    gamma.scale = parameters.beta_gamma * ones(T, 1);    
    omega.shape = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.scale = parameters.beta_omega * ones(P, 1);
    epsilon.shape = (parameters.alpha_epsilon + 0.5 * N) .* ones(T, 1);
    epsilon.scale = parameters.beta_epsilon * ones(T, 1);
    be.mean = [zeros(T, 1); ones(P, 1)];
    be.covariance = eye(T + P, T + P);

    KmKm = cell(1, T);
    for t = 1:T
        KmKm{t} = zeros(D(t), D(t));
        for m = 1:P
            KmKm{t} = KmKm{t} + Km{t}(:, :, m) * Km{t}(:, :, m)';
        end
        Km{t} = reshape(Km{t}, [D(t), N(t) * P]);
    end
    
    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT = cell(1, T);
    for t = 1:T
        atimesaT{t}.mean = a{t}.mean * a{t}.mean' + a{t}.covariance;
    end
    GtimesGT = cell(1, T);
    for t = 1:T
        GtimesGT{t}.mean = G{t}.mean * G{t}.mean' + N(t) * G{t}.covariance;
    end
    btimesbT.mean = be.mean(1:T) * be.mean(1:T)' + be.covariance(1:T, 1:T);
    etimeseT.mean = be.mean(T + 1:T + P) * be.mean(T + 1:T + P)' + be.covariance(T + 1:T + P, T + 1:T + P);
    etimesb.mean = zeros(P, T);
    for t = 1:T
        etimesb.mean(:, t) = be.mean(T + 1:T + P) * be.mean(t) + be.covariance(T + 1:T + P, t);
    end
    KmtimesGT = cell(1, T);
    for t = 1:T
        KmtimesGT{t}.mean = Km{t} * reshape(G{t}.mean', N(t) * P, 1);
    end
    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update lambda
        for t = 1:T
            lambda{t}.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT{t}.mean));
        end
        %%%% update upsilon
        for t = 1:T
            upsilon.scale(t) = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT{t}.mean)) ...
                                                                         - 2 * sum(sum(reshape(a{t}.mean' * Km{t}, [N(t), P])' .* G{t}.mean)) ...
                                                                         + sum(diag(KmKm{t} * atimesaT{t}.mean))));
        end
        %%%% update a
        for t = 1:T
            a{t}.covariance = (diag(lambda{t}.shape .* lambda{t}.scale) + upsilon.shape(t) * upsilon.scale(t) * KmKm{t}) \ eye(D(t), D(t));
            a{t}.mean = a{t}.covariance * (upsilon.shape(t) * upsilon.scale(t) * KmtimesGT{t}.mean);
            atimesaT{t}.mean = a{t}.mean * a{t}.mean' + a{t}.covariance;
        end
        %%%% update G        
        for t = 1:T
            G{t}.covariance = (upsilon.shape(t) * upsilon.scale(t) * eye(P, P) + epsilon.shape(t) * epsilon.scale(t) * etimeseT.mean) \ eye(P, P);
            G{t}.mean = G{t}.covariance * (upsilon.shape(t) * upsilon.scale(t) * reshape(a{t}.mean' * Km{t}, [N(t), P])' + epsilon.shape(t) * epsilon.scale(t) * (be.mean(T + 1:T + P) * y{t}' - repmat(etimesb.mean(:, t), 1, N(t))));
            GtimesGT{t}.mean = G{t}.mean * G{t}.mean' + N(t) * G{t}.covariance;
            KmtimesGT{t}.mean = Km{t} * reshape(G{t}.mean', N(t) * P, 1);
        end   
        %%%% update gamma
        gamma.scale = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mean));
        %%%% update omega
        omega.scale = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mean));
        %%%% update epsilon
        for t = 1:T
            epsilon.scale(t) = 1 / (1 / parameters.beta_epsilon + 0.5 * (y{t}' * y{t} - 2 * y{t}' * [ones(1, N(t)); G{t}.mean]' * be.mean([t, T + 1:T + P]) ...
                                                                         + N(t) * btimesbT.mean(t, t) ...
                                                                         + sum(diag(GtimesGT{t}.mean * etimeseT.mean)) ...
                                                                         + 2 * sum(G{t}.mean, 2)' * etimesb.mean(:, t)));
        end
        %%%% update b and e
        be.covariance = [diag(gamma.shape .* gamma.scale) + diag(N .* epsilon.shape .* epsilon.scale), zeros(T, P); ...
                         zeros(P, T), diag(omega.shape .* omega.scale)];
        for t = 1:T
            be.covariance(T + 1:T + P, t) = epsilon.shape(t) * epsilon.scale(t) * sum(G{t}.mean, 2);
            be.covariance(t, T + 1:T + P) = epsilon.shape(t) * epsilon.scale(t) * sum(G{t}.mean, 2)';
            be.covariance(T + 1:T + P, T + 1:T + P) = be.covariance(T + 1:T + P, T + 1:T + P) + epsilon.shape(t) * epsilon.scale(t) * GtimesGT{t}.mean;
        end
        be.covariance = be.covariance \ eye(T + P, T + P);
        be.mean = zeros(T + P, 1);        
        for t = 1:T
            be.mean(t) = epsilon.shape(t) * epsilon.scale(t) * sum(y{t});
            be.mean(T + 1:T + P) = be.mean(T + 1:T + P) + epsilon.shape(t) * epsilon.scale(t) * G{t}.mean * y{t};
        end
        be.mean = be.covariance * be.mean;
        btimesbT.mean = be.mean(1:T) * be.mean(1:T)' + be.covariance(1:T, 1:T);
        etimeseT.mean = be.mean(T + 1:T + P) * be.mean(T + 1:T + P)' + be.covariance(T + 1:T + P, T + 1:T + P);
        for t = 1:T
            etimesb.mean(:, t) = be.mean(T + 1:T + P) * be.mean(t) + be.covariance(T + 1:T + P, t);
        end
        
        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            for t = 1:T
                lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda{t}.shape) + log(lambda{t}.scale)) ...
                              - lambda{t}.shape .* lambda{t}.scale / parameters.beta_lambda ...
                              - gammaln(parameters.alpha_lambda) ...
                              - parameters.alpha_lambda * log(parameters.beta_lambda));
            end
            %%%% p(upsilon)
            lb = lb + sum((parameters.alpha_upsilon - 1) * (psi(upsilon.shape) + log(upsilon.scale)) ...
                          - upsilon.shape .* upsilon.scale / parameters.beta_upsilon ...
                          - gammaln(parameters.alpha_upsilon) ...
                          - parameters.alpha_upsilon * log(parameters.beta_upsilon));
            %%%% p(a | lambda)
            for t = 1:T
                lb = lb - 0.5 * sum(diag(diag(lambda{t}.shape .* lambda{t}.scale) * atimesaT{t}.mean)) ...
                        - 0.5 * (D(t) * log2pi - sum(log(lambda{t}.shape .* lambda{t}.scale)));
            end
            %%%% p(G | a, Km, upsilon)
            for t = 1:T
                lb = lb - 0.5 * sum(diag(GtimesGT{t}.mean)) * upsilon.shape(t) * upsilon.scale(t) ...
                        + (a{t}.mean' * KmtimesGT{t}.mean) * upsilon.shape(t) * upsilon.scale(t) ...
                        - 0.5 * sum(diag(KmKm{t} * atimesaT{t}.mean)) * upsilon.shape(t) * upsilon.scale(t) ...
                        - 0.5 * N(t) * P * (log2pi - log(upsilon.shape(t) * upsilon.scale(t)));
            end
            %%%% p(gamma)
            lb = lb + sum((parameters.alpha_gamma - 1) * (psi(gamma.shape) + log(gamma.scale)) ...
                          - gamma.shape .* gamma.scale / parameters.beta_gamma ...
                          - gammaln(parameters.alpha_gamma) ...
                          - parameters.alpha_gamma * log(parameters.beta_gamma));
            %%%% p(b | gamma)
            lb = lb - 0.5 * sum(diag(diag(gamma.shape .* gamma.scale) * btimesbT.mean)) ...
                    - 0.5 * (T * log2pi - sum(log(gamma.shape .* gamma.scale)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.shape) + log(omega.scale)) ...
                          - omega.shape .* omega.scale / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(diag(diag(omega.shape .* omega.scale) * etimeseT.mean)) ...
                    - 0.5 * (P * log2pi - sum(log(omega.shape .* omega.scale)));
            %%%% p(epsilon)
            lb = lb + sum((parameters.alpha_epsilon - 1) * (psi(epsilon.shape) + log(epsilon.scale)) ...
                          - epsilon.shape .* epsilon.scale / parameters.beta_epsilon ...
                          - gammaln(parameters.alpha_epsilon) ...
                          - parameters.alpha_epsilon * log(parameters.beta_epsilon));
            %%%% p(y | b, e, G, epsilon)
            for t = 1:T
                lb = lb - 0.5 * (y{t}' * y{t}) * epsilon.shape(t) * epsilon.scale(t) ...
                        + (y{t}' * (G{t}.mean' * be.mean(T + 1:T + P))) * epsilon.shape(t) * epsilon.scale(t) ...
                        + sum(be.mean(t) * y{t}) * epsilon.shape(t) * epsilon.scale(t) ...
                        - 0.5 * sum(diag(etimeseT.mean * GtimesGT{t}.mean)) * epsilon.shape(t) * epsilon.scale(t) ...
                        - sum(G{t}.mean' * etimesb.mean(:, t)) * epsilon.shape(t) * epsilon.scale(t) ...
                        - 0.5 * N(t) * btimesbT.mean(t, t) * epsilon.shape(t) * epsilon.scale(t) ...
                        - 0.5 * N(t) * (log2pi - log(epsilon.shape(t) * epsilon.scale(t)));
            end

            %%%% q(lambda)
            for t = 1:T
                lb = lb + sum(lambda{t}.shape + log(lambda{t}.scale) + gammaln(lambda{t}.shape) + (1 - lambda{t}.shape) .* psi(lambda{t}.shape));
            end
            %%%% q(upsilon)
            lb = lb + sum(upsilon.shape + log(upsilon.scale) + gammaln(upsilon.shape) + (1 - upsilon.shape) .* psi(upsilon.shape));            
            %%%% q(a)
            for t = 1:T
                lb = lb + 0.5 * (D(t) * (log2pi + 1) + logdet(a{t}.covariance));
            end
            %%%% q(G)
            for t = 1:T
                lb = lb + 0.5 * N(t) * (P * (log2pi + 1) + logdet(G{t}.covariance));
            end
            %%%% q(gamma)
            lb = lb + sum(gamma.shape + log(gamma.scale) + gammaln(gamma.shape) + (1 - gamma.shape) .* psi(gamma.shape));
            %%%% q(omega)
            lb = lb + sum(omega.shape + log(omega.scale) + gammaln(omega.shape) + (1 - omega.shape) .* psi(omega.shape));
            %%%% q(epsilon)
            lb = lb + sum(epsilon.shape + log(epsilon.scale) + gammaln(epsilon.shape) + (1 - epsilon.shape) .* psi(epsilon.shape));
            %%%% q(b, e)
            lb = lb + 0.5 * ((T + P) * (log2pi + 1) + logdet(be.covariance));

            bounds(iter) = lb;
        end
    end

    state.lambda = lambda;
    state.upsilon = upsilon;
    state.a = a;
    state.gamma = gamma;
    state.omega = omega;
    state.epsilon = epsilon;
    state.be = be;
    if parameters.progress == 1
        state.bounds = bounds;
    end
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
	ld = 2 * sum(log(diag(U)));
end
