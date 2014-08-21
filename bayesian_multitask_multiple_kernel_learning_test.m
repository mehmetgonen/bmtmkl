% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bayesian_multitask_multiple_kernel_learning_test(Km, state)
    T = length(Km);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(Km{t}, 2);
    end
    P = size(Km{1}, 3);

    prediction.G = cell(1, T);
    for t = 1:T
        prediction.G{t}.mean = zeros(P, N(t));
        prediction.G{t}.covariance = zeros(P, N(t));
        for m = 1:P
            prediction.G{t}.mean(m, :) = state.a{t}.mean' * Km{t}(:, :, m);
            prediction.G{t}.covariance(m, :) = 1 / (state.upsilon.shape(t) * state.upsilon.scale(t)) + diag(Km{t}(:, :, m)' * state.a{t}.covariance * Km{t}(:, :, m));
        end
    end
    
    prediction.y = cell(1, T);
    for t = 1:T
        prediction.y{t}.mean = [ones(1, N(t)); prediction.G{t}.mean]' * state.be.mean([t, T + 1:T + P]);
        prediction.y{t}.covariance = 1 / (state.epsilon.shape(t) * state.epsilon.scale(t)) + diag([ones(1, N(t)); prediction.G{t}.mean]' * state.be.covariance([t, T + 1:T + P], [t, T + 1:T + P]) * [ones(1, N(t)); prediction.G{t}.mean]);
    end
end
