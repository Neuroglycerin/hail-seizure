function featReduceDimCSP(outfeatname, featnames, modtyps, subj, ictypgroupings, ncomponents, removebest)

% Input handling ----------------------------------------------------------
if nargin<7
    removebest = false;
end
if ncomponents<0;
    error('ncomponents must be positive');
end

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';
settings = json.read(settingsfname);
featversion = settings.VERSION;

allictyps = {'preictal'; 'interictal'; 'test'; 'pseudopreictal'; 'pseudointerictal';};

% Main --------------------------------------------------------------------
% Compute covariance matrices
fprintf('--Computing covariance matrices for %s--\n',subj);
[S, segnames] = computeCov(featnames, modtyps, subj, ictypgroupings, featversion);

% Compute CSP.
% NB: W will work on data which is shaped [components, samples]
fprintf('--Computing CSP projection weights for %s--\n',subj);
W = csp(S{1},S{2});
clear S;

% Reduce the size of the output by removing output components from W
if ncomponents<1
    ncomponents = floor(size(W,1)*ncomponents);
    if ~removebest || mod(size(W,1),2)==0
        % Round to nearest even number
        ncomponents = ncomponents + mod(ncomponents,2);
    end
end
if ncomponents>size(W,1)
    warning('Too many components requested: %d (%d available)',ncomponents,size(W,1));
    ncomponents = size(W,1);
end
if ~removebest
    W = W(1:ncomponents, :);
else
    W = W((size(W,1)-ncomponents+1):end, :);
end

% Save W for posterity
fprintf('--Saving weights for %s to output file--\n',subj);
saveWeightsToFile();

% Use W to project to this space and save to file
h5fnme = getFeatH5fname(outfeatname, 'combo', featversion);
fprintf('--Saving transformed features to output file %s--\n',h5fnme);
saveComboFeat(h5fnme, W, featnames, modtyps, subj, allictyps, featversion);


    function saveWeightsToFile()
        % Write the CSP weights matrix to file ------------------------------------
        [Wfnamefull,Wfnamefull_log] = getWfname(outfeatname, subj);
        fprintf('Writing weight matrix to file\n  %s\n',Wfnamefull);
        if ~exist(fileparts(Wfnamefull),'dir'); mkdir(fileparts(Wfnamefull)); end;
        save(Wfnamefull, '-v7.3', 'W', 'ncomponents', 'removebest');
        if ~exist(fileparts(Wfnamefull_log),'dir'); mkdir(fileparts(Wfnamefull_log)); end;
        % Save a dated copy for posterity
        save(Wfnamefull_log, '-v7.3', 'W', 'ncomponents', 'removebest');
    end

end


function [Wfnamefull,Wfnamefull_log] = getWfname(outfeatname, subj)

% Declarations
settingsfname = 'SETTINGS.json';

% Load the settings file
settings = json.read(settingsfname);

mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['cspcombodimreduce_' outfeatname '_' subj];

Wfnamefull = fullfile(mydir,Wfname);
Wfnamefull_log = fullfile(mydir,'log',[Wfname '_' datestr(now,30)]);

end


function S = computeCovHIGHMEMORY(subj, ictyps, featnames, modtyps, featversion)

loadascell = true;

% Input handling ----------------------------------------------------------
if ~isequal(size(modtyps),size(featnames))
    error('Feature sizes mismatch');
end
if numel(ictyps)~=2
    error('CSP needs to work on two conditions');
end

% Initialise
S = cell(size(ictyps));

for iCondition=1:2
    allfeat = [];
    for iIct = 1:numel(ictyps{iCondition})
        ictfeat = [];
        for iFtr=1:numel(featnames)
            % Load this feature for this ictyp
            [featM, fnms] = getFeatFromHDF5(featnames{iFtr}, subj, ictyps{iCondition}{iIct}, modtyps{iFtr}, featversion, loadascell);
            if iFtr==1
                targetfnms = fnms;
            elseif ~isequal(fnms,targetfnms)
                error('Inconsistent segment file names');
            end
            % Flatten
            featM = reshape(featM,size(featM,1),[]);
            % Add to the holding matrix
            ictfeat(:,end+(1:size(featM,2))) = featM;
        end
        % Add each ictyp together
        allfeat(end+(1:size(ictfeat,1)),:) = ictfeat;
    end
    S{iCondition} = allfeat * allfeat';
    S{iCondition} = S{iCondition} / trace(S{iCondition});
end

end


function [S, segnames] = computeCov(featnames, modtyps, subj, ictyps, featversion)

% Input handling ----------------------------------------------------------
if ~isequal(size(modtyps),size(featnames))
    error('Feature sizes mismatch');
end
if numel(ictyps)~=2
    error('CSP needs to work on two conditions');
end

% Main --------------------------------------------------------------------
% Initialise
S = cell(size(ictyps));

% Get metadata structure
Info = h5info(getFeatH5fname(featnames{1}, modtyps{1}, featversion));

for iCondition=1:2
    for iIct = 1:numel(ictyps{iCondition})
        % Need to get a list of all the datasets
        segnames = scrapeH5datasets(Info, subj, ictyps{iCondition}{iIct}, true);
        for iSeg=1:numel(segnames)
            % Load the full feature vector for this segname
            % fprintf('%s\n',segnames{iSeg})
            allfeatV = loadFullFeatV(featnames, modtyps, subj, ictyps{iCondition}{iIct}, segnames{iSeg}, featversion, iSeg==1);
            % Initialise covariance matrix
            if iIct==1 && iSeg==1
                S{iCondition} = zeros(size(allfeatV,2));
            end
            % Take covariance estimate for this segment
            S{iCondition} = S{iCondition} + allfeatV' * allfeatV;
        end
    end
    S{iCondition} = S{iCondition} / trace(S{iCondition});
end

end

function allfeatV = loadFullFeatV(featnames, modtyps, subj, ictyp, segname, featversion, dbgmd)

if nargin<7;
    dbgmd = false;
end

% Initialise
allfeatV = [];
for iFtr=1:numel(featnames)
    if dbgmd; fprintf('Loading %s %s\n',modtyps{iFtr},featnames{iFtr}); end;
    % Work out h5 filename
    h5fnme = getFeatH5fname(featnames{iFtr}, modtyps{iFtr}, featversion);
    % Check file exists
    if ~exist(h5fnme,'file');
        warning('getFeatFromHDF5:NoFile','HDF5 file %s does not exist',h5fnme);
        % error('getFeatFromHDF5:NoFile','HDF5 file %s does not exist',h5fnme);
    end
    
    % Load the feature for this segment
    featV = h5read(h5fnme, ['/' subj '/' ictyp '/' segname]);
    
    % Reshape so it is vector-like
    featV = reshape(featV,size(featV,1),[]);
    % Expanding vector! Bad practice for memory consumption!
    allfeatV(:,end+(1:size(featV,2))) = featV;
    
end

end


function saveComboFeat(outfname, W, featnames, modtyps, subj, ictyps, featversion)

% Need to take transpose
W = W';

% Get metadata structure
Info = h5info(getFeatH5fname(featnames{1}, modtyps{1}, featversion));

% Loop over every segment
for iIct = 1:numel(ictyps)
    % Need to get a list of all the datasets
    segnames = scrapeH5datasets(Info, subj, ictyps{iIct}, true);
    for iSeg=1:numel(segnames)
        % Load the full feature vector for this segname
        allfeatV = loadFullFeatV(featnames, modtyps, subj, ictyps{iIct}, segnames{iSeg}, featversion);
        % Apply the basis transformation
        allfeatV = allfeatV * W;
        % Save it to the output HDF5
        dataset = strcat('/', subj, '/', ictyps{iIct}, '/', segnames{iSeg});
        h5writePlus(outfname, dataset, allfeatV);
    end
end

end


function W = csp(S1,S0)

if size(S1,1)~=size(S1,2)
    error('Covariance matrices should be square');
end
if ~isequal(size(S1),size(S0))
    error('Covariance matrices should be same size');
end

nChn = size(S1,1);

% Taken from RegCsp
%Equation (7) in the paper
SigmaComps = S0 + S1; 
[Ucomps,lmds] = eig(SigmaComps);
[lmds,Idxs] = sort(diag(lmds),'descend');
Ucomps = Ucomps(:,Idxs);

%Note equations (8) to (12) in the conference version is now condensed in
%one equation (8) in the journal version
%Equation (8) in the CONFERENCE paper
P = sqrt(inv(diag(lmds)))*Ucomps';

%Equation (9) in CONFERENCE the paper
Sgm1 = P*S1*P';

%Equation (11) in the CONFERENCE paper
[B,D] = eig(Sgm1);
[D,Idxs] = sort(diag(D),'descend'); 
B = B(:,Idxs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure; plot(D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Equation (12) in the CONFERENCE paper
%Equation (8) in the JOURNAL paper
W=(B'*P); 
%Normalize the projrection matrix
for i=1:length(Idxs), W(i,:)=W(i,:)./norm(W(i,:)); end

% Sort columns, take first and last columns first, etc
idx = zeros(1,nChn);
for iChn=1:floor(nChn/2)
    idx(iChn*2-1) = iChn;
    idx(iChn*2  ) = nChn-iChn+1;
end
if mod(nChn,2)==1; idx(end) = ceil(nChn/2); end;
W = W(idx,:);

end