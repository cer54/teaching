function trainData(X,y,N_train,eta,maxit,nl,l)
%TRAINDATA trains a logistic classification model on data and produces some
%measures of the performance
%   trainData(X,y,N_train,eta,maxit,nl,l) takes in a data matrix X where
%   each data point corresponds to a row and the corresponding labels are
%   stored in y. N_train gives the size of the training set, eta is the
%   learnung rate, maxit gives the number of iterations to run the
%   optimizer for, nl indicates whether linear or RBFs are used and
%   l gives the length scale for these RBFs. nl = 1 for RBFs and nl = 0 for
%   linear features
    
    N = size(X,1);   %No. of data points
    
    train_labels = y(1:N_train);
    test_labels = y(N_train+1:end);
    N_test = length(test_labels);
    
    if nl == 1
        train_data = [ones(N_train,1) zeros(N_train,N_train)];
        test_data = [ones(N_test,1) zeros(N_test,N_train)];
        for i=1:N_train
            train_data(:,i+1) = sum((X(1:N_train,:) - repmat(X(i,:),N_train,1)).^2,2);
            test_data(:,i+1) = sum((X(N_train+1:end,:) - repmat(X(i,:),N_test,1)).^2,2);
        end
    
        train_data = exp(-(0.5*(l^-2))*train_data);
        test_data = exp(-(0.5*(l^-2))*test_data);
    else
        train_data = [ones(N_train,1) X(1:N_train,:)];
        test_data = [ones(N-N_train,1) X(N_train+1:end,:)];
    end
    
   
    
    %Randomly initialize parameters from Normal distribution
    beta_init = randn(size(train_data,2),1);
    
    
    %Grid parameters
    x1_min = -3; x1_max = 3; x2_min = -3; x2_max = 3;
    x1 = linspace(x1_min,x1_max);
    x2 = linspace(x2_min,x2_max);
    [X1,X2] = meshgrid(x1,x2);
    
    num_iters = 0;
    %First iteration
    beta_prev = beta_init;
    grad = train_data'*(train_labels-sigmoid(train_data*beta_prev));% - beta_prev;
    beta = beta_prev + eta*grad;
    num_iters = num_iters+1;
    
    L_train = logLikelihood(train_data,train_labels,beta);
    L_test = logLikelihood(test_data,test_labels,beta);
    subplot(1,2,1);
    plot(L_train,'rx');
    hold on;
    plot(L_test,'bo');
    legend('Training','Tests')
    ylabel('$log P(y|X,\beta)$','Interpreter','latex');
    xlabel('No. of itereations','Interpreter','latex');
    subplot(1,2,2);
    contour(X1,X2,zfunc(X1,X2,l,X(1:N_train,:),beta,nl));
    hold on;
    scatter(X(:,1),X(:,2),[],y);
    hold off;
    pause
        
    %Perform gradient ascent for given number of iterations
    while num_iters < maxit
        beta_prev = beta;
        grad = train_data'*(train_labels-sigmoid(train_data*beta_prev));%-beta_prev;
        beta = beta_prev + eta*grad;
        
        %norm(beta-beta_prev)
        %Plot training curves and contour plot
        L_train = [L_train logLikelihood(train_data,train_labels,beta)];
        L_test = [L_test logLikelihood(test_data,test_labels,beta)];
        if mod(num_iters,20) == 0
            subplot(1,2,1);
            plot(L_train,'rx');
            hold on;
            plot(L_test,'bo');
            legend('Training','Tests')
            ylabel('$log P(y|X,\beta)$','Interpreter','latex');
            xlabel('No. of itereations','Interpreter','latex');
            subplot(1,2,2);
            contour(X1,X2,zfunc(X1,X2,l,X(1:N_train,:),beta,nl));
            hold on;
            scatter(X(:,1),X(:,2),[],y);
            hold off;
            pause
        end
        num_iters = num_iters+1;
    end
    
    disp(['Log-likelihood (train) per data point: ',num2str(logLikelihood(train_data,train_labels,beta)/N_train)]);
    disp(['Log-likelihood (test) per data point: ',num2str(logLikelihood(test_data,test_labels,beta)/N_test)]);
    
    false_positives = [];
    true_positives = [];
    for tau=0:0.01:1
        predicted_labels = sigmoid(test_data*beta) > tau;
        num_negatives = sum(test_labels==0);
        num_positives = N_test - num_negatives;
    
        true_negatives = sum(predicted_labels(test_labels==0)==0)/num_negatives;
        false_positives = [false_positives sum(predicted_labels(test_labels==0)==1)/num_negatives];
        false_negatives = sum(predicted_labels(test_labels==1)==0)/num_positives;
        true_positives = [true_positives sum(predicted_labels(test_labels==1)==1)/num_positives];
    end
    
     figure;
     plot(false_positives,true_positives);
     xlabel('$P(\hat{y}=1|y=0)$','interpreter','latex')
     ylabel('$P(\hat{y}=1|y=1)$','interpreter','latex')
     trapz((0:0.01:1),true_positives)
    
%           disp(['True negatives: ',num2str(true_negatives)])
%           disp(['False positives: ',num2str(false_positives)])
%           disp(['False negatives: ',num2str(false_negatives)])
%           disp(['True positives: ',num2str(true_positives)])

end

function [sig_z] = sigmoid(z)
    sig_z = 1./(1+exp(-z));
end

function [L] = logLikelihood(X,y,beta)
    L = y'*log(sigmoid(X*beta)) + (1-y)'*log(sigmoid(-X*beta));
end

function [Z] = zfunc(X1,X2,l,features,beta,nl)

    if nl == 1
        grid = [reshape(X1,size(X1,2)*size(X1,1),1) reshape(X2,size(X1,2)*size(X1,1),1)];
        num_pts = size(grid,1);
        N_train = size(features,1);
    
        Z_features = [ones(num_pts,1) zeros(num_pts,N_train)];
        for i=1:N_train
            Z_features(:,i+1) = sum((grid - repmat(features(i,:),num_pts,1)).^2,2);     
        end
    
        Z_features = exp(-(0.5*(l^-2))*Z_features);
    
        Z = sigmoid(reshape(Z_features*beta,size(X1)));
    else
        Z = sigmoid(beta(1)+beta(2)*X1+beta(3)*X2);
    end
    
end