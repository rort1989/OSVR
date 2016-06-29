%% example of training and testing OSVR for expression intensity estimation
clear all; close all;

%% load data
% train_data_seq: an array of cells containing training feature sequences,
% each cell contains a D*T matrix where D is dimension of feature and T is
% the sequence length
% train_label_seq: an array of cells containing training intensity labels
% for all the sequences, each cell contains a K*2 matrix where K is the
% number of frames with labeled intensities. The first column is the index
% of frame and the second column is associated intensity value
% test_data: a D*T' matrix containing testing frames, where D is the
% dimension of feature and T' is number of testing frames
load('data.mat','train_data_seq','train_label_seq','test_data','test_label');

%% define constant
loss = 2; % loss function of OSVR
bias = 1; % include bias term or not in OSVR
lambda = 1; % scaling parameter for primal variables in OSVR
gamma = [100 1]; % loss balance parameter
smooth = 1; % temporal smoothness on ordinal constraints
epsilon = [0.1 1]; % parameter in epsilon-SVR
rho = 0.1; % augmented Lagrangian multiplier
flag = 0; % unsupervise learning flag
max_iter = 300; % maximum number of iteration in optimizating OSVR

%% Training 
% formalize coefficients data structure
[A,c,D,nInts,nPairs,weight] = constructParams(train_data_seq,train_label_seq,epsilon,bias,flag);
mu = gamma(1)*ones(nInts+nPairs,1); % change the values if you want to assign different weights to different samples
mu(nInts+1:end) = gamma(2)/gamma(1)*mu(nInts+1:end);
if smooth % add temporal smoothness
    mu = mu.*weight;
end
% solve the OSVR optimization problem in ADMM
[model,history,z] = admm(A,c,lambda,mu,'option',loss,'rho',rho,'max_iter',max_iter,'bias',1-bias); % 
theta = model.w;
    
%% Testing 
% perform testing
dec_values =theta'*[test_data; ones(1,size(test_data,2))];
% compute evaluation metrics
RR = corrcoef(dec_values,test_label);  
ee = dec_values - test_label; 
dat = [dec_values; test_label]'; 
ry_test = RR(1,2); % Pearson Correlation Coefficient (PCC)
abs_test = sum(abs(ee))/length(ee); % Mean Absolute Error (MAE)
mse_test = ee(:)'*ee(:)/length(ee); % Mean Square Error (MSE)
icc_test = ICC(3,'single',dat); % Intra-Class Correlation (ICC)

%% Visualize results
plot(test_label); hold on; 
plot(dec_values,'r');
legend('Ground truth','Prediction')
