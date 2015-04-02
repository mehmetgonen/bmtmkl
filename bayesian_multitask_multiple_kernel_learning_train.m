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
        lambda{t}.alpha = (parameters.alpha_lambda + 0.5) * ones(D(t), 1);
        lambda{t}.beta = parameters.beta_lambda * ones(D(t), 1);
    end
    upsilon.alpha = (parameters.alpha_upsilon + 0.5 * N * P) .* ones(T, 1);
    upsilon.beta = parameters.beta_upsilon * ones(T, 1);
    a = cell(1, T);    
    for t = 1:T
        a{t}.mu = randn(D(t), 1);
        a{t}.sigma = eye(D(t), D(t));
    end
    G = cell(1, T);
    for t = 1:T
        G{t}.mu = randn(P, N(t));
        G{t}.sigma = eye(P, P);
    end
    gamma.alpha = (parameters.alpha_gamma + 0.5) * ones(T, 1);
    gamma.beta = parameters.beta_gamma * ones(T, 1);    
    omega.alpha = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.beta = parameters.beta_omega * ones(P, 1);
    epsilon.alpha = (parameters.alpha_epsilon + 0.5 * N) .* ones(T, 1);
    epsilon.beta = parameters.beta_epsilon * ones(T, 1);
    be.mu = [zeros(T, 1); ones(P, 1)];
    be.sigma = eye(T + P, T + P);

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
        atimesaT{t}.mu = a{t}.mu * a{t}.mu' + a{t}.sigma;
    end
    GtimesGT = cell(1, T);
    for t = 1:T
        GtimesGT{t}.mu = G{t}.mu * G{t}.mu' + N(t) * G{t}.sigma;
    end
    btimesbT.mu = be.mu(1:T) * be.mu(1:T)' + be.sigma(1:T, 1:T);
    etimeseT.mu = be.mu(T + 1:T + P) * be.mu(T + 1:T + P)' + be.sigma(T + 1:T + P, T + 1:T + P);
    etimesb.mu = zeros(P, T);
    for t = 1:T
        etimesb.mu(:, t) = be.mu(T + 1:T + P) * be.mu(t) + be.sigma(T + 1:T + P, t);
    end
    KmtimesGT = cell(1, T);
    for t = 1:T
        KmtimesGT{t}.mu = Km{t} * reshape(G{t}.mu', N(t) * P, 1);
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
            lambda{t}.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT{t}.mu));
        end
        %%%% update upsilon
        for t = 1:T
            upsilon.beta(t) = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT{t}.mu)) ...
                                                                         - 2 * sum(sum(reshape(a{t}.mu' * Km{t}, [N(t), P])' .* G{t}.mu)) ...
                                                                         + sum(diag(KmKm{t} * atimesaT{t}.mu))));
        end
        %%%% update a
        for t = 1:T
            a{t}.sigma = (diag(lambda{t}.alpha .* lambda{t}.beta) + upsilon.alpha(t) * upsilon.beta(t) * KmKm{t}) \ eye(D(t), D(t));
            a{t}.mu = a{t}.sigma * (upsilon.alpha(t) * upsilon.beta(t) * KmtimesGT{t}.mu);
            atimesaT{t}.mu = a{t}.mu * a{t}.mu' + a{t}.sigma;
        end
        %%%% update G        
        for t = 1:T
            G{t}.sigma = (upsilon.alpha(t) * upsilon.beta(t) * eye(P, P) + epsilon.alpha(t) * epsilon.beta(t) * etimeseT.mu) \ eye(P, P);
            G{t}.mu = G{t}.sigma * (upsilon.alpha(t) * upsilon.beta(t) * reshape(a{t}.mu' * Km{t}, [N(t), P])' + epsilon.alpha(t) * epsilon.beta(t) * (be.mu(T + 1:T + P) * y{t}' - repmat(etimesb.mu(:, t), 1, N(t))));
            GtimesGT{t}.mu = G{t}.mu * G{t}.mu' + N(t) * G{t}.sigma;
            KmtimesGT{t}.mu = Km{t} * reshape(G{t}.mu', N(t) * P, 1);
        end   
        %%%% update gamma
        gamma.beta = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mu));
        %%%% update omega
        omega.beta = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mu));
        %%%% update epsilon
        for t = 1:T
            epsilon.beta(t) = 1 / (1 / parameters.beta_epsilon + 0.5 * (y{t}' * y{t} - 2 * y{t}' * [ones(1, N(t)); G{t}.mu]' * be.mu([t, T + 1:T + P]) ...
                                                                         + N(t) * btimesbT.mu(t, t) ...
                                                                         + sum(diag(GtimesGT{t}.mu * etimeseT.mu)) ...
                                                                         + 2 * sum(G{t}.mu, 2)' * etimesb.mu(:, t)));
        end
        %%%% update b and e
        be.sigma = [diag(gamma.alpha .* gamma.beta) + diag(N .* epsilon.alpha .* epsilon.beta), zeros(T, P); ...
                         zeros(P, T), diag(omega.alpha .* omega.beta)];
        for t = 1:T
            be.sigma(T + 1:T + P, t) = epsilon.alpha(t) * epsilon.beta(t) * sum(G{t}.mu, 2);
            be.sigma(t, T + 1:T + P) = epsilon.alpha(t) * epsilon.beta(t) * sum(G{t}.mu, 2)';
            be.sigma(T + 1:T + P, T + 1:T + P) = be.sigma(T + 1:T + P, T + 1:T + P) + epsilon.alpha(t) * epsilon.beta(t) * GtimesGT{t}.mu;
        end
        be.sigma = be.sigma \ eye(T + P, T + P);
        be.mu = zeros(T + P, 1);        
        for t = 1:T
            be.mu(t) = epsilon.alpha(t) * epsilon.beta(t) * sum(y{t});
            be.mu(T + 1:T + P) = be.mu(T + 1:T + P) + epsilon.alpha(t) * epsilon.beta(t) * G{t}.mu * y{t};
        end
        be.mu = be.sigma * be.mu;
        btimesbT.mu = be.mu(1:T) * be.mu(1:T)' + be.sigma(1:T, 1:T);
        etimeseT.mu = be.mu(T + 1:T + P) * be.mu(T + 1:T + P)' + be.sigma(T + 1:T + P, T + 1:T + P);
        for t = 1:T
            etimesb.mu(:, t) = be.mu(T + 1:T + P) * be.mu(t) + be.sigma(T + 1:T + P, t);
        end
        
        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            for t = 1:T
                lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda{t}.alpha) + log(lambda{t}.beta)) ...
                              - lambda{t}.alpha .* lambda{t}.beta / parameters.beta_lambda ...
                              - gammaln(parameters.alpha_lambda) ...
                              - parameters.alpha_lambda * log(parameters.beta_lambda));
            end
            %%%% p(upsilon)
            lb = lb + sum((parameters.alpha_upsilon - 1) * (psi(upsilon.alpha) + log(upsilon.beta)) ...
                          - upsilon.alpha .* upsilon.beta / parameters.beta_upsilon ...
                          - gammaln(parameters.alpha_upsilon) ...
                          - parameters.alpha_upsilon * log(parameters.beta_upsilon));
            %%%% p(a | lambda)
            for t = 1:T
                lb = lb - 0.5 * sum(diag(diag(lambda{t}.alpha .* lambda{t}.beta) * atimesaT{t}.mu)) ...
                        - 0.5 * (D(t) * log2pi - sum(psi(lambda{t}.alpha) + log(lambda{t}.beta)));
            end
            %%%% p(G | a, Km, upsilon)
            for t = 1:T
                lb = lb - 0.5 * sum(diag(GtimesGT{t}.mu)) * upsilon.alpha(t) * upsilon.beta(t) ...
                        + (a{t}.mu' * KmtimesGT{t}.mu) * upsilon.alpha(t) * upsilon.beta(t) ...
                        - 0.5 * sum(diag(KmKm{t} * atimesaT{t}.mu)) * upsilon.alpha(t) * upsilon.beta(t) ...
                        - 0.5 * N(t) * P * (log2pi - (psi(upsilon.alpha(t)) + log(upsilon.beta(t))));
            end
            %%%% p(gamma)
            lb = lb + sum((parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) ...
                          - gamma.alpha .* gamma.beta / parameters.beta_gamma ...
                          - gammaln(parameters.alpha_gamma) ...
                          - parameters.alpha_gamma * log(parameters.beta_gamma));
            %%%% p(b | gamma)
            lb = lb - 0.5 * sum(diag(diag(gamma.alpha .* gamma.beta) * btimesbT.mu)) ...
                    - 0.5 * (T * log2pi - sum(psi(gamma.alpha) + log(gamma.beta)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.alpha) + log(omega.beta)) ...
                          - omega.alpha .* omega.beta / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(diag(diag(omega.alpha .* omega.beta) * etimeseT.mu)) ...
                    - 0.5 * (P * log2pi - sum(psi(omega.alpha) + log(omega.beta)));
            %%%% p(epsilon)
            lb = lb + sum((parameters.alpha_epsilon - 1) * (psi(epsilon.alpha) + log(epsilon.beta)) ...
                          - epsilon.alpha .* epsilon.beta / parameters.beta_epsilon ...
                          - gammaln(parameters.alpha_epsilon) ...
                          - parameters.alpha_epsilon * log(parameters.beta_epsilon));
            %%%% p(y | b, e, G, epsilon)
            for t = 1:T
                lb = lb - 0.5 * (y{t}' * y{t}) * epsilon.alpha(t) * epsilon.beta(t) ...
                        + (y{t}' * (G{t}.mu' * be.mu(T + 1:T + P))) * epsilon.alpha(t) * epsilon.beta(t) ...
                        + sum(be.mu(t) * y{t}) * epsilon.alpha(t) * epsilon.beta(t) ...
                        - 0.5 * sum(diag(etimeseT.mu * GtimesGT{t}.mu)) * epsilon.alpha(t) * epsilon.beta(t) ...
                        - sum(G{t}.mu' * etimesb.mu(:, t)) * epsilon.alpha(t) * epsilon.beta(t) ...
                        - 0.5 * N(t) * btimesbT.mu(t, t) * epsilon.alpha(t) * epsilon.beta(t) ...
                        - 0.5 * N(t) * (log2pi - (psi(epsilon.alpha(t)) + log(epsilon.beta(t))));
            end

            %%%% q(lambda)
            for t = 1:T
                lb = lb + sum(lambda{t}.alpha + log(lambda{t}.beta) + gammaln(lambda{t}.alpha) + (1 - lambda{t}.alpha) .* psi(lambda{t}.alpha));
            end
            %%%% q(upsilon)
            lb = lb + sum(upsilon.alpha + log(upsilon.beta) + gammaln(upsilon.alpha) + (1 - upsilon.alpha) .* psi(upsilon.alpha));            
            %%%% q(a)
            for t = 1:T
                lb = lb + 0.5 * (D(t) * (log2pi + 1) + logdet(a{t}.sigma));
            end
            %%%% q(G)
            for t = 1:T
                lb = lb + 0.5 * N(t) * (P * (log2pi + 1) + logdet(G{t}.sigma));
            end
            %%%% q(gamma)
            lb = lb + sum(gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) .* psi(gamma.alpha));
            %%%% q(omega)
            lb = lb + sum(omega.alpha + log(omega.beta) + gammaln(omega.alpha) + (1 - omega.alpha) .* psi(omega.alpha));
            %%%% q(epsilon)
            lb = lb + sum(epsilon.alpha + log(epsilon.beta) + gammaln(epsilon.alpha) + (1 - epsilon.alpha) .* psi(epsilon.alpha));
            %%%% q(b, e)
            lb = lb + 0.5 * ((T + P) * (log2pi + 1) + logdet(be.sigma));

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
