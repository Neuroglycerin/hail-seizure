%% Matlab script to pre-process raw data and output serialized features

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

subj = subjnames();
ictypes = {'preictal'; 'interictal'; 'test'};
feature_funcs = {...
    @feat_var; ...
    @feat_cov; ...
    @feat_corrcoef; ...
    @feat_pib; ...
    @feat_xcorr; ...
    @feat_psd; ...
    @feat_psd_logf;
    @feat_coher;
    @feat_coher_logf};
           
for i = 1:numel(ictypes)
    for j = 1:numel(feature_funcs)
        getFeatAddToHDF5(feature_funcs{j}, subj{1}, ictypes{i}, 'raw')
    end
end
