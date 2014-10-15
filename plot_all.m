
subj = subjnames();

modtyps = {'raw','ica'};

featurenames = {...
    'feat_var'; ...
    'feat_cov'; ...
    'feat_corrcoef'; ...
    'feat_pib'; ...
    'feat_psd'; ...
    'feat_psd_logf'; ...
    'feat_coher'; ...
    'feat_coher_logf'; ...
    'feat_xcorr';
    'feat_act';
    'feat_corrcoefeig';
    'feat_FFT';
    'feat_FFTcorrcoef';
    'feat_FFTcorrcoefeig';
    'feat_pib_ratio';
    'feat_pib_ratioBB';
    'feat_PSDcorrcoef';
    'feat_PSDcorrcoefeig';
    'feat_PSDlogfcorrcoef';
    'feat_PSDlogfcorrcoefeig';
    };

%%
for iFtr=1:length(featurenames) % 1:length(featurenames)
    for iMod=1:length(modtyps)
        for iSub=1:length(subj) % 1:length(subj)
            fprintf('%s: Plotting %s %s %s\n',datestr(now,30),featurenames{iFtr}, subj{iSub}, modtyps{iMod});
            plot_feat(featurenames{iFtr}, subj{iSub}, modtyps{iMod});
        end
    end
end

%%
for iFtr=1:length(featurenames) % 1:length(featurenames)
    for iMod=1:length(modtyps)
        fprintf('%s: Plotting %s %s\n',datestr(now,30),featurenames{iFtr}, modtyps{iMod});
        plot_info(featurenames{iFtr},modtyps{iMod});
        close all;
    end
end