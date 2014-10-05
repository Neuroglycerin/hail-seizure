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


for iFun = 1:numel(feature_funcs)
    for iSub=1:numel(subj)
        for iIct = 1:numel(ictypes)
            fprintf('Running feature %s on raw %s %s\n',func2str(feature_funcs{iFun}), subj{iSub}, ictypes{iIct});
            getFeatAddToHDF5(feature_funcs{iFun}, subj{iSub}, ictypes{iIct}, 'raw')
        end
    end
end
