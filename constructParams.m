function [A,c,D,num_intensity,num_pairs,weight] = constructParams(dataset,labelset,epsilon,bias,flag)
% formalize model parameters used by admm solver for OSVR

if ~iscell(dataset) % only one sequence
    datacells{1} = dataset;
    labelcells{1} = labelset;
else % multiple sequences
    datacells = dataset;
    labelcells = labelset;
end
N = numel(datacells); 
T = zeros(N,1);
num_pairs_max = 0;
num_intensity = 0;
% collect statistics of the dataset
for n = 1:N
    [D,T(n)] = size(datacells{n});
    num_pairs_max = num_pairs_max + T(n)*(T(n)+1)/2;
    num_intensity = num_intensity + 2*size(labelcells{n},1);
end
% initialize the components for OSVR problem
% pre-allocate storage for A and e for efficiency
A = zeros(num_intensity+num_pairs_max,D+bias);
c = ones(num_intensity+num_pairs_max,1);
weight = ones(num_intensity+num_pairs_max,1);
idx_row_I = 0;
idx_row_P = num_intensity;
num_pairs = 0;
for n = 1:N
    data = datacells{n};
    label = labelcells{n};  
    nframe = size(label,1);
    peak = max(label(:,2)); % index of apex frame
    idx = find(label(:,2)==peak); % all the indices with peak intensity
    apx = label(idx(max(1,ceil(length(idx)/2))),1); % select apx to be the median one of all the peak frames
    % based on apex frame, create the ordinal set
    % number of ordinal pair
    pairs = zeros(T(n)*(T(n)+1)/2,2);
    dist = ones(T(n)*(T(n)+1)/2,1);
    count = 0;
    for i = apx:-1:2
        pairs(count+1:count+i-1,1) = i;
        pairs(count+1:count+i-1,2) = [i-1:-1:1]';
        dist(count+1:count+i-1) = [1:i-1]';
        count = count + i-1;    
    end
    if apx < T(n)
        for i = apx:T(n)       
            pairs(count+1:count+T(n)-i,1) = i;
            pairs(count+1:count+T(n)-i,2) = [i+1:T(n)]';
            dist(count+1:count+T(n)-i) = [1:T(n)-i]';
            count = count + T(n)-i;
        end
    end
    pairs = pairs(1:count,:);
    dist = dist(1:count);
    num_pairs = num_pairs + count;
    % compute objective function value and gradient of objective function
    dat = data(:,label(:,1)); % D*num_labels
    tij = data(:,pairs(:,1)) - data(:,pairs(:,2)); % D*num_pairs
    % assign values to parameters
    A(idx_row_I+1:idx_row_I+nframe,1:D) = dat';
    A(idx_row_I+1+num_intensity/2:idx_row_I+nframe+num_intensity/2,1:D) = -dat';
    A(idx_row_P+1:idx_row_P+count,1:D) = -tij';
    c(idx_row_I+1:idx_row_I+nframe) = -epsilon(1)*ones(nframe,1) - label(:,2);
    c(idx_row_I+1+num_intensity/2:idx_row_I+nframe+num_intensity/2) = -epsilon(1)*ones(nframe,1) + label(:,2);
    c(idx_row_P+1:idx_row_P+count) = epsilon(2);
    % weights for ordinal loss is inverse proportial to the distance between the pair of frames
    weight(idx_row_P+1:idx_row_P+count) = 1./dist;
    idx_row_I = idx_row_I + nframe;
    idx_row_P = idx_row_P + count;
end
% truncate to the actual number of rows
A = A(1:num_intensity+num_pairs,:);
if bias % augment A for including bias term
    A(1:num_intensity/2,D+1) = 1;
    A(1+num_intensity/2:num_intensity,D+1) = -1;
end
c = c(1:num_intensity+num_pairs,:);
weight = weight(1:num_intensity+num_pairs);
% unsupervisd flag to exclude all the rows associated with intensity lables
if flag
    A = A(num_intensity+1:end,:);
    c = c(num_intensity+1:end,:);
    weight = weight(num_intensity+1:end,:);
    num_intensity = 0;
end

end