function estimateParams(train_data,train_labels,test_data,test_labels,eta,thresh)
%ESTIMATEPARAMS estimates the parameters of a model through gradient ascent
%   [beta_est] = estimateParams(X,y,beta_init,eta,thresh) takes in a data
%   matrix X of shape [no of features, no of data points] that has labels,
%   y, of size 'no of data points' and performs gradient ascent to estimate
%   the parameters for a logistic classification model. eta is the learning 
%   rate and thresh is the threshold at which gradient ascent is cut off.
    
    train_size = length(train_labels);
    test_size = length(test_labels);
    
    %Randomly initialize parameters from Normal distribution
    beta_init = randn(size(train_data,2),1);
    
    %norm(y-X*beta_init)
    %Grid parameters
    x_min = -3; x_max = 3; y_min = -3; y_max = 3;
    x = linspace(x_min,x_max);
    y = linspace(y_min,y_max);
    [X,Y] = meshgrid(x,y);
    
    %First iteration
    beta_prev = beta_init;
    grad = train_data'*(train_labels-sigmoid(train_data*beta_prev));
    beta = beta_prev + eta*grad;
    L_train = logLikelihood(train_data,train_labels,beta);
    L_test = logLikelihood(test_data,test_labels,beta);
    
    %Perform gradient ascent until the change in parameters is small enough
    while norm(beta-beta_prev) > thresh
        beta_prev = beta;
        grad = train_data'*(train_labels-sigmoid(train_data*beta_prev));
        beta = beta_prev + eta*grad;
        
        %Plot training curves and contour plot
        L_train = [L_train logLikelihood(train_data,train_labels,beta)];
        L_test = [L_test logLikelihood(test_data,test_labels,beta)];
        subplot(1,2,1);
        plot(L_train,'rx');
        hold on;
        plot(L_test,'bo');
        subplot(1,2,2);
        contour(X,Y,sigmoid(beta(0)+beta(1)*X+beta(2)*Y));
        pause
    end
    
    %norm(y-X*beta)
    
    %Return the final estimated parameters
    %beta_est = beta;
end

function [sig_z] = sigmoid(z)
    sig_z = 1./(1+exp(-z));
end

function [L] = logLikelihood(X,y,beta)
    L = y'*log(sigmoid(X*beta)) + (1-y)'*log(sigmoid(-X*beta));
end