function [logp] = logmvnpdf(x,mu,Sigma, pi_k)

[D,~] = size(x);
const = -0.5 * D * log(2*3.1416);

term1 = -0.5 * ((x - mu)' * (inv(Sigma) * (x - mu)));
term2 = - 0.5 * logdet(Sigma);    
logp = (const + term1 + term2) +  log(pi_k);
% logp = logGaussPdf(x, mu, Sigma) * pi;
end

function y = logdet(A)
 y = log(det(A));
end

function y = logGaussPdf(X, mu, sigma)
    % Compute log pdf of a Gaussian distribution.
    % Written by Mo Chen (mochen80@gmail.com).
    [d,k] = size(mu);

    if size(sigma,1)==d && size(sigma,2)==d && k==1
        X = bsxfun(@minus,X,mu);
        [R,p]= chol(sigma);
        if p ~= 0
            error('ERROR: sigma is not PD.');
        end
        Q = R'\X;
        q = dot(Q,Q,1);  % quadratic term (M distance)
        c = d*log(2*pi)+2*sum(log(diag(R)));   % normalization constant
        y = -(c+q)/2;
    elseif size(sigma,1)==d && size(sigma,2)==k
        lambda = 1./sigma;
        ml = mu.*lambda;
        q = bsxfun(@plus,X'.^2*lambda-2*X'*ml,dot(mu,ml,1)); % M distance
        c = (d*log(2*pi)+sum(log(sigma),1))/(-2); % normalization constant
        y = bsxfun(@plus,q/(-2),c);
    elseif size(sigma,1)==1 && size(sigma,2)==k
        X2 = repmat(dot(X,X,1)',1,k);
        D = bsxfun(@plus,X2-2*X'*mu,dot(mu,mu,1));
        q = bsxfun(@times,D,1./sigma);  % M distance
        c = d*log(2*pi*sigma)/(-2);          % normalization constant
        y = bsxfun(@plus,q/(-2),c);
    else
        error('Parameters mismatched.');
    end
end