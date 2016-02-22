%% Matlab script to pre-process raw data and output serialized features
% nice matlab13 -nodisplay -nosplash -r "preprocess_feat_missing; exit;"

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

logfname = fullfile('train', ...
    sprintf('featureprocessinglog-%s.txt',datestr(now,30)));

subjname_list = subjnames();

% modtyps = {'raw','ica'};
% modtyps = {'ica'};
% modtyps = {'csp'};
% modtyps = {'cln,raw,dwn','ica,raw,dwn','cln,csp,dwn'};
% modtyps = {'cln,raw,dwn'};
modtyps = {
%    'cln,ica,dwn';
%    'cln,csp,dwn';
    'cln,raw,dwn';
};

% ictypes = {'preictal'; 'interictal'; 'test'};
% ictypes = {'pseudopreictal'; 'pseudointerictal';};
ictypes = {'preictal'; 'interictal'; 'test'; 'pseudopreictal'; 'pseudointerictal';};

nSplits_list = [1 10];

feature_funcs = {
    @feat_gcaus;
};

matlabpool('local',12);

for nSplits = nSplits_list(:)'
for iMod = 1:numel(modtyps)
modtyp = modtyps{iMod};
for iFun = 1:numel(feature_funcs)
    tic2 = tic;
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: starting to compute %d %s on %s\n', ...
        datestr(now,30), getComputerName(), nSplits, func2str(feature_funcs{iFun}), modtyp);
    fclose(fid);
    for iSub=1:numel(subjname_list)
        for iIct = 1:numel(ictypes)
            fprintf('%s %s: Running feature %d %s on %s %s %s\n', ...
                getComputerName(), datestr(now,30), nSplits, func2str(feature_funcs{iFun}), modtyp, subjname_list{iSub}, ictypes{iIct});
            tic1 = tic;
            getFeatAddToHDF5(feature_funcs{iFun}, subjname_list{iSub}, ictypes{iIct}, modtyp, nSplits)
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
	% computeInfoAddToHDF5(myfeat, subjname_list{iSub}, modtyp);
    end
    tme = toc(tic2);
    tme = tme/60/60;
    hrs = floor(tme);
    mins = (tme-hrs)*60;
    secs = round(mod(mins,1)*60);
    mins = floor(mins);
    fprintf('Feature %d %s took %d h %d m %d s \n', ...
        nSplits, func2str(feature_funcs{iFun}), hrs, mins, secs);
    fid = fopen(logfname,'a');
    fprintf(fid,'%s %s: %d %s %s took %.1f seconds\n', ...
        datestr(now,30), getComputerName(), nSplits, func2str(feature_funcs{iFun}), modtyp, toc(tic2));
    fclose(fid);
end
end
end

% % Close workers
% myCluster = parcluster('local');
% delete(myCluster.Jobs);

% Close pool
poolobj = gcp('nocreate');
delete(poolobj);
