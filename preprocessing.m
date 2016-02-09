% Preprocess all the data

% -----------------------------------------------------------------------------
% Add all the functions to the MATLAB path
addpath(genpath('matlab'));

% Define epsilon for ICA
epsilon = 1e-12;

% Get the list of subject names we will need to loop over
subjname_list = subjnames();

% -----------------------------------------------------------------------------
% Clean the data
preprocess_clean;

% -----------------------------------------------------------------------------
% Do signal decomposition for each subject

% Compute ICA weights
extramodtyp = 'cln,dwn';
for i=1:length(subjname_list);
    computeSubjICAW(subjname_list{i}, extramodtyp, epsilon);
end

% Compute CSP weights
extramodtyp = 'cln,dwn';
for i=1:length(subjname_list);
    fprintf('%s: Computing CSP weights for %s %s\n', ...
        datestr(now,30), extramodtyp, subjname_list{i});
    saveCSPweights(subjname_list{i}, extramodtyp);
end

% Compute ICA weights to use after CSP with dimensionality reduction
extramodtyp = 'cln,cspdr,dwn';
for i=1:length(subjname_list);
    computeSubjICAW(subjname_list{i}, extramodtyp, epsilon);
end

% -----------------------------------------------------------------------------
% Generate all the features
preprocess_feat;
