load ps6_data
params = InitParams1;

NUM_CLUSTERS = size(params.mu, 2);
NUM_DATA = size(Spikes, 2);
DIMENSION = size(params.mu, 1);

% Initializing and plotting all the spikes
N_k = zeros(1, NUM_CLUSTERS);
covvar = repmat(params.Sigma, [1,1,NUM_CLUSTERS]);
for i=1:NUM_DATA
    plot(Spikes(:,i))
    hold on
end
figure;

for k=1:100
    % E Step : Choose between mvnpdf and custom implementation
    r = zeros(NUM_CLUSTERS, NUM_DATA);
    for i=1:NUM_CLUSTERS
        for j=1:NUM_DATA
%              r(i, j) = mvnpdf(Spikes(:,j), params.mu(:,i), ...
%                  covvar(:,:,i)) * params.pi(i);
             r(i,j) = exp(logmvnpdf(Spikes(:,j),params.mu(:,i), ...
                      covvar(:,:,i))-boxed_term(Spikes(:,j), params, ...
                      covvar)) * params.pi(i);            
        end
    end
    r_sum = sum(r);
    for i=1:NUM_CLUSTERS
        r(i,:) = r(i,:)./r_sum;  
    end
    r_log = -log(r);


    % M Step
    for i=1:NUM_CLUSTERS
        mean_sum = 0;
        N_k = zeros(1, NUM_CLUSTERS);
        for j=1:NUM_DATA
            mean_sum = mean_sum + r(i,j).* Spikes(:,j);
            N_k(i) = N_k(i) + r(i,j);
        end
        params.mu(:,i) = mean_sum ./ N_k(i);
    end

    for i=1:NUM_CLUSTERS
        cov_sum = 0;
        for j=1:NUM_DATA
            cov_sum = cov_sum + r(i,j).*((Spikes(:,j) - params.mu(:,i)) * (Spikes(:,j) - params.mu(:,i))');
        end
        covvar(:,:,i) = cov_sum;
    end

    % Evaluation log likelihood
    L = 0;
    for i =1:NUM_DATA
        L_this = 0;
        for j = 1:NUM_CLUSTERS
%             L_this = L_this + mvnpdf(Spikes(:,i), params.mu(:,j), ...
%                 covvar(:,:,j)) * params.pi(j);
            L_this = L_this + exp(logmvnpdf(Spikes(:,i),params.mu(:,j), ...
                      covvar(:,:,j))-boxed_term(Spikes(:,i), params, ...
                      covvar)) * params.pi(j); 
        end
        L = L + L_this;
    end
    L
end


