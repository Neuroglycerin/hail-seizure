%% Matlab script to pre-process raw data and output serialized features

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

subj = {'Dog_5'};
ictypes = {'preictal'; 'interictal'; 'test'};
feature_funcs = {@feat_cov; ...
                 @feat_psd_logf; ...
                 @feat_var};
           
for i = 1:numel(ictypes)
    for j = 1:numel(feature_funcs)
        getFeatAddToHDF5(feature_funcs{j}, subj{1}, ictypes{i}, 'raw')
    end
end
