%% Matlab script to pre-process raw data and output serialized features

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

logfname = fullfile('train','featureprocessinglog.txt');

subj = subjnames();

modtyps = {'raw','ica'};

% ictypes = {'preictal'; 'interictal'; 'test'};
ictypes = {'pseudopreictal'; 'pseudointerictal';};

nSplits = 1;

feature_funcs = {
    @feat_var;
    @feat_cov;
    @feat_corrcoef;
	@feat_corrcoefeig;
    @feat_pib;
    @feat_pib_ratioBB;
    @feat_pib_ratio;
    @feat_psd_logf;
    @feat_coher_logf;
    @feat_act;
    @feat_xcorr;
	@feat_PSDlogfcorrcoef;
	@feat_PSDlogfcorrcoefeig
    @feat_psd;
    @feat_coher;
	@feat_PSDcorrcoef;
	@feat_PSDcorrcoefeig;
    @feat_FFT;
    @feat_FFTcorrcoef;
    @feat_FFTcorrcoefeig;
    };

feature_funcs = {
    @feat_pib;
    @feat_pib_ratioBB;
    @feat_pib_ratio;
    @feat_psd_logf;
    @feat_coher_logf;
    @feat_act;
    @feat_xcorr;
	@feat_PSDlogfcorrcoef;
	@feat_PSDlogfcorrcoefeig
    @feat_psd;
    @feat_coher;
	@feat_PSDcorrcoef;
	@feat_PSDcorrcoefeig;
    @feat_FFT;
    @feat_FFTcorrcoef;
    @feat_FFTcorrcoefeig;
    };

matlabpool('local',12);

for iMod = 1:numel(modtyps)
modtyp = modtyps{iMod};
for iFun = 1:numel(feature_funcs)
    tic2 = tic;
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: starting to compute %s on %s\n',datestr(now,30),getComputerName(),func2str(feature_funcs{iFun}), modtyp);
    fclose(fid);
    for iSub=1:numel(subj)
        for iIct = 1:numel(ictypes)
            fprintf('%s %s: Running feature %s on %s %s %s\n', getComputerName(), datestr(now,30), func2str(feature_funcs{iFun}), modtyp, subj{iSub}, ictypes{iIct});
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
	computeInfoAddToHDF5([num2str(nSplits) func2str(feature_funcs{iFun})], subj{iSub}, modtyp);
    end
    tme = toc(tic2);
    tme = tme/60/60;
    hrs = floor(tme);
    mins = (tme-hrs)*60;
    secs = round(mod(mins,1)*60);
    mins = floor(mins);
    fprintf('Feature %s took %d h %d m %d s \n',func2str(feature_funcs{iFun}),hrs,mins,secs);
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: %s %s took %.1f seconds\n',datestr(now,30),getComputerName(),func2str(feature_funcs{iFun}), modtyp, toc(tic2));
    fclose(fid);
end
end

% % Close workers
% myCluster = parcluster('local');
% delete(myCluster.Jobs);

% Close pool
poolobj = gcp('nocreate');
delete(poolobj);
