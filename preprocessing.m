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
    tic2 = tic;
    for iSub=1:numel(subj)
        for iIct = 1:numel(ictypes)
            fprintf('%s: Running feature %s on raw %s %s\n', datestr(now,30), func2str(feature_funcs{iFun}), subj{iSub}, ictypes{iIct});
            tic1 = tic;
            getFeatAddToHDF5(feature_funcs{iFun}, subj{iSub}, ictypes{iIct}, 'raw')
            tme = toc(tic1);
            tme = tme/60/60;
            hrs = floor(tme);
            mins = (tme-hrs)*60;
            secs = round(mod(mins,1)*60);
            mins = floor(mins);
            fprintf('took %d h %d m %d s \n',hrs,mins,secs);
        end
    end
    tme = toc(tic2);
    tme = tme/60/60;
    hrs = floor(tme);
    mins = (tme-hrs)*60;
    secs = round(mod(mins,1)*60);
    mins = floor(mins);
    fprintf('Feature %s took %d h %d m %d s \n',func2str(feature_funcs{iFun}),hrs,mins,secs);
end
