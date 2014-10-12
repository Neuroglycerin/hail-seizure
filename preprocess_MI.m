
settingsfname = 'SETTINGS.json';
settings = json.read(settingsfname);
featversion = settings.VERSION;

modtyps = {'raw';'ica'};

subj = subjnames();

featurenames = {...
    'feat_var'; ...
    'feat_cov'; ...
    'feat_corrcoef'; ...
    'feat_pib'; ...
    'feat_psd'; ...
    'feat_psd_logf'; ...
    'feat_coher'; ...
    'feat_coher_logf'; ...
    'feat_xcorr';};

parfor iSet = 1:length(modtyps)*length(featurenames)
    iFtr = mod((iSet-1),length(featurenames))+1;
    %iSub = mod(floor((iSet-1)/length(featurenames)),length(subj))+1;
    %iMod = floor((iSet-1)/length(featurenames)/length(subj))+1;
    iMod = floor((iSet-1)/length(featurenames))+1;
    %fprintf('%d %d %d\n',iFtr,iSub,iMod);
    for iSub=1:length(subj)
        try
            computeInfoAddToHDF5(featurenames{iFtr}, subj{iSub}, modtyps{iMod}, featversion);
        catch ME
            fprintf('Error for %s %s %s: %s\n', ...
                subj{iSub}, modtyps{iMod}, featurenames{iFtr}, ME.message);
        end
    end
end