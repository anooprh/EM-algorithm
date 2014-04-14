function [logp] = logmvnpdf(x,mu,Sigma)
% outputs log likelihood array for observations x  where x_n ~ N(mu,Sigma)
% x is NxD, mu is 1xD, Sigma is DxD

[D,N] = size(x);
const = -0.5 * D * log(2*pi);

term1 = -0.5 * ((x - mu)' * (Sigma \ (x - mu))); % N x 1
term2 = const - 0.5 * logdet(Sigma);    % scalar
logp = term1' + term2;

end

function y = logdet(A)

U = chol(A);
y = 2*sum(log(diag(U)));

end