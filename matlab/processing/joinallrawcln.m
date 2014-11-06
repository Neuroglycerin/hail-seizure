
% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

feature_funcs = {
    'feat_var'
    'feat_cov'
    'feat_lmom-1'
    'feat_lmom-2'
    'feat_lmom-3'
    'feat_lmom-4'
    'feat_lmom-5'
    'feat_lmom-6'
    'feat_corrcoef'
    'feat_corrcoefeig'
    'feat_pib'
    'feat_pib_ratioBB'
    'feat_pib_ratio'
    'feat_psd_logf'
    'feat_coher_logf'
    'feat_act'
    'feat_xcorr-ypeak'
    'feat_xcorr-tpeak'
    'feat_xcorr-twidth'
    'feat_PSDlogfcorrcoef'
    'feat_PSDlogfcorrcoefeig'
    'feat_PSDcorrcoef'
    'feat_PSDcorrcoefeig'
    'feat_FFT'
    'feat_FFTcorrcoef'
    'feat_FFTcorrcoefeig'
    'feat_pwling4'
    'feat_pwling5'
    'feat_pwling2'
    'feat_pwling1'
    'feat_ilingam-B'
    'feat_ilingam-k'
    'feat_psd'
    'feat_coher'
    'feat_phase-theta-dif'
    'feat_phase-theta-sync'
    'feat_phase-delta-dif'
    'feat_phase-delta-sync'
    'feat_phase-alpha-dif'
    'feat_phase-alpha-sync'
    'feat_phase-beta-dif'
    'feat_phase-beta-sync'
    'feat_phase-low_gamma-dif'
    'feat_phase-low_gamma-sync'
    'feat_phase-high_gamma-dif'
    'feat_phase-high_gamma-sync'
    'feat_mvar-ARF'
    'feat_mvar-COHphs'
    'feat_mvar-COH'
    'feat_mvar-DCphs'
    'feat_mvar-DC'
    'feat_mvar-DTFphs'
    'feat_mvar-DTF'
    'feat_mvar-GPDCphs'
    'feat_mvar-GPDC'
    'feat_mvar-Hphs'
    'feat_mvar-H'
    'feat_mvar-PCOHphs'
    'feat_mvar-PCOH'
    'feat_mvar-PDCphs'
    'feat_mvar-PDC'
    'feat_mvar-Pphs'
    'feat_mvar-P'
    'feat_mvar-Sphs'
    'feat_mvar-S'
    };

modtyps = {'raw'}; %,'ica','csp'};

nParts = 10;

for iMod=1:numel(modtyps)
    featisgood = zeros(size(feature_funcs));
    for iFtr=1:numel(feature_funcs)
        feat_str = feature_funcs{iFtr};
        if ~ischar(feat_str)
            feat_str = func2str(feat_str);
            feature_funcs{iFtr} = feat_str;
        end
        if nParts~=1
            feat_str = [num2str(nParts) feat_str];
        end
        modtyp = modtyps{iMod};
        fprintf('Copying %s into cln,%s,dwn %s\n',modtyp,modtyp,feat_str);
        try
            joinrawcln(feat_str, modtyp, '_v2');
            featisgood(iFtr) = 1;
        catch ME
            if strcmp(ME.identifier,'getFeatFromHDF5:NoFile')
                fprintf('\t No file for %s %s\n',modtyp,feat_str);
                featisgood(iFtr) = 0;
            elseif strcmp(ME.identifier,'joinrawcln:NoRawFile')
                fprintf('\t No file for %s %s\n',modtyp,feat_str);
                featisgood(iFtr) = -2;
            elseif strcmp(ME.identifier,'joinrawcln:NoClnFile')
                fprintf('\t No file for cln,%s,dwn %s\n',modtyp,feat_str);
                featisgood(iFtr) = -3;
            elseif strcmp(ME.identifier,'getFeatFromHDF5:missingSubject')
                fprintf('\t %s for %s\n',ME.message,feat_str);
                featisgood(iFtr) = -1;
            elseif strcmp(ME.identifier,'getFeatFromHDF5:missingIctal')
                fprintf('\t %s for %s\n',ME.message,feat_str);
                featisgood(iFtr) = -1;
            else
                rethrow(ME);
            end
            continue;
        end
        fprintf('Successfully copied %s into cln,%s,dwn %s\n',modtyp,modtyp,feat_str);
    end
    fprintf('------------------\n');
    fprintf('Successful features:\n  ');
    fprintf('%s, ',feature_funcs{featisgood==1});
    fprintf('\n------------------\n');
    fprintf('Unsuccessful features due to missing %s file:\n  ',modtyp);
    fprintf('%s, ',feature_funcs{featisgood==-2});
    fprintf('\n------------------\n');
    fprintf('Unsuccessful features due to missing cln,%s,dwn file:\n  ',modtyp);
    fprintf('%s, ',feature_funcs{featisgood==-3});
    fprintf('\n------------------\n');
    fprintf('Unsuccessful features due to missing datasets:\n  ');
    fprintf('%s, ',feature_funcs{featisgood==-1});
    fprintf('\n------------------\n');
    fprintf('Otherwise unsuccessful features:\n  ');
    fprintf('%s, ',feature_funcs{featisgood==0});
    fprintf('\n------------------\n');
end