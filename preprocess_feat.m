%% Matlab script to pre-process raw data and output serialized features

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

logfname = fullfile('train',sprintf('featureprocessinglog-%s.txt',datestr(now,30)));

subj = subjnames();
subj = {'Patient_2'};

% modtyps = {'raw','ica'};
% modtyps = {'ica'};
% modtyps = {'csp'};
% modtyps = {'cln,raw,dwn','ica,raw,dwn','cln,csp,dwn'};
% modtyps = {'cln,raw,dwn'};
modtyps = {'cln,csp,dwn'};

% ictypes = {'preictal'; 'interictal'; 'test'};
% ictypes = {'pseudopreictal'; 'pseudointerictal';};
ictypes = {'preictal'; 'interictal'; 'test'; 'pseudopreictal'; 'pseudointerictal';};

nSplits = 10;

feature_funcs = {
    @feat_var;
    @feat_cov;
    @feat_lmom;
    @feat_corrcoef;
    @feat_corrcoefeig;
    @feat_pib_ratioBB;
    @feat_mvar;
    @feat_phase;
    @feat_ampcorrcoef
    @feat_psd_logf;
    @feat_coher_logf;
    @feat_pib
    @feat_pib_ratio
    @feat_act;
    @feat_xcorr;
    @feat_PSDlogfcorrcoef;
    @feat_PSDlogfcorrcoefeig;
    @feat_pwling4
    @feat_spearman
    @feat_PSDcorrcoef;
    @feat_PSDcorrcoefeig;
    @feat_FFTcorrcoef;
    @feat_FFTcorrcoefeig;
    @feat_pwling5;
    @feat_pwling2;
    @feat_pwling1;
    @feat_ilingam;
    @feat_emvar
    @feat_FFT;
    @feat_psd;
    @feat_coher;
    };

matlabpool('local',12);

for iMod = 1:numel(modtyps)
modtyp = modtyps{iMod};
for iFun = 1:numel(feature_funcs)
    tic2 = tic;
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: starting to compute %d %s on %s\n',datestr(now,30),getComputerName(),nSplits,func2str(feature_funcs{iFun}), modtyp);
    fclose(fid);
    for iSub=1:numel(subj)
        for iIct = 1:numel(ictypes)
            fprintf('%s %s: Running feature %d %s on %s %s %s\n', getComputerName(), datestr(now,30), nSplits, func2str(feature_funcs{iFun}), modtyp, subj{iSub}, ictypes{iIct});
            tic1 = tic;
            getFeatAddToHDF5(feature_funcs{iFun}, subj{iSub}, ictypes{iIct}, modtyp, nSplits)
            tme = toc(tic1);
            tme = tme/60/60;
            hrs = floor(tme);
            mins = (tme-hrs)*60;
            secs = round(mod(mins,1)*60);
            mins = floor(mins);
            fprintf('took %d h %d m %d s \n',hrs,mins,secs);
        end
	% Add mutual information to HDF5 as well
        myfeat = func2str(feature_funcs{iFun});
        if nSplits~=1
            myfeat = [num2str(nSplits) myfeat];
        end
	% computeInfoAddToHDF5(myfeat, subj{iSub}, modtyp);
    end
    tme = toc(tic2);
    tme = tme/60/60;
    hrs = floor(tme);
    mins = (tme-hrs)*60;
    secs = round(mod(mins,1)*60);
    mins = floor(mins);
    fprintf('Feature %d %s took %d h %d m %d s \n',nSplits,func2str(feature_funcs{iFun}),hrs,mins,secs);
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: %d %s %s took %.1f seconds\n',datestr(now,30),getComputerName(),nSplits,func2str(feature_funcs{iFun}), modtyp, toc(tic2));
    fclose(fid);
end
end

% % Close workers
% myCluster = parcluster('local');
% delete(myCluster.Jobs);

% Close pool
poolobj = gcp('nocreate');
delete(poolobj);
