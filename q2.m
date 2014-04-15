load ps6_data
params = InitParams1;

NUM_CLUSTERS = size(params.mu, 2);
NUM_DATA = size(Spikes, 2);
DIMENSION = size(params.mu, 1);

% Initializing and plotting all the spikes
N_Iter = 100;
N_k = zeros(1, NUM_CLUSTERS);
likelihood = zeros(1, N_Iter);
covvar = repmat(params.Sigma, [1,1,NUM_CLUSTERS]);
for i=1:NUM_DATA
    plot(Spikes(:,i))
    hold on
end
figure;

for k=1:N_Iter
    % E Step : Choose between mvnpdf and custom implementation
    r = zeros(NUM_CLUSTERS, NUM_DATA);
    for i=1:NUM_DATA
        bt = boxed_term(Spikes(:,i), params, covvar);
        for j=1:NUM_CLUSTERS
             r(j,i) = exp(logmvnpdf(Spikes(:,i), params.mu(:,j), ...
                 covvar(:,:,j), params.pi(j)) - bt);
        end
    end
   
    % M Step
    for i=1:NUM_CLUSTERS
         N_k(i) = sum(r(i,:));
    end
    for i=1:NUM_CLUSTERS
        mean_sum = 0;      
        for j=1:NUM_DATA
            mean_sum = mean_sum + r(i,j).* Spikes(:,j);
        end
        params.mu(:,i) = mean_sum ./ N_k(i);
        params.pi(i) = N_k(i) / NUM_DATA;
    end

    for i=1:NUM_CLUSTERS
        cov_sum = 0;
        for j=1:NUM_DATA
            cov_sum = cov_sum + r(i,j).*((Spikes(:,j) - params.mu(:,i)) * (Spikes(:,j) - params.mu(:,i))');
        end
        covvar(:,:,i) = cov_sum ./ N_k(i);
    end

    % Evaluation log likelihood
    L = 0;
    for i=1:NUM_DATA
        L = L + boxed_term(Spikes(:,i), params, covvar);
    end
    likelihood(k) = L;
end
plot(likelihood);

