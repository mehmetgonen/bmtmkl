function prediction = bayesian_multitask_multiple_kernel_learning_test(Km, state)
    T = length(Km);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(Km{t}, 2);
    end
    P = size(Km{1}, 3);

    prediction.G = cell(1, T);
    for t = 1:T
        prediction.G{t}.mu = zeros(P, N(t));
        prediction.G{t}.sigma = zeros(P, N(t));
        for m = 1:P
            prediction.G{t}.mu(m, :) = state.a{t}.mu' * Km{t}(:, :, m);
            prediction.G{t}.sigma(m, :) = 1 / (state.upsilon.alpha(t) * state.upsilon.beta(t)) + diag(Km{t}(:, :, m)' * state.a{t}.sigma * Km{t}(:, :, m));
        end
    end
    
    prediction.y = cell(1, T);
    for t = 1:T
        prediction.y{t}.mu = [ones(1, N(t)); prediction.G{t}.mu]' * state.be.mu([t, T + 1:T + P]);
        prediction.y{t}.sigma = 1 / (state.epsilon.alpha(t) * state.epsilon.beta(t)) + diag([ones(1, N(t)); prediction.G{t}.mu]' * state.be.sigma([t, T + 1:T + P], [t, T + 1:T + P]) * [ones(1, N(t)); prediction.G{t}.mu]);
    end
end
