function W = saveCSPweights(subj, modtyp)
% You should pre-multiply by W

if nargin<2 || isempty(modtyp)
    modtyp = '';
end

dbgmde = true;

% Compute CSP weights for this subject
W = computeCSP(subj, modtyp);

% Write the CSP weights matrix to file ------------------------------------
[Wfname,Wfnamefull_log] = getWfname(subj, modtyp);
if dbgmde; fprintf('Writing weight matrix to file\n  %s\n',Wfname); end
if ~exist(fileparts(Wfname),'dir'); mkdir(fileparts(Wfname)); end;
save(Wfname,'W'); % Overwrite the copy to be used in transformations
if ~exist(fileparts(Wfnamefull_log),'dir'); mkdir(fileparts(Wfnamefull_log)); end;
save(Wfnamefull_log,'W'); % Save a dated copy for posterity

end

function [Wfnamefull,Wfnamefull_log] = getWfname(subj, modtyp)

% Declarations
settingsfname = 'SETTINGS.json';

% Load the settings file
settings = json.read(settingsfname);

if isempty(modtyp) || strcmp(modtyp,'raw')
    modtyp = '';
else
    modtyp = ['_' modtyp];
end

mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['csp_weights_' subj modtyp];

Wfnamefull = fullfile(mydir,Wfname);
Wfnamefull_log = fullfile(mydir,'log',[Wfname '_' datestr(now,30)]);

end

function W = computeCSP(subj, modtyp)

if isempty(modtyp) || strcmp(modtyp,'raw')
    modtyp = 'raw';
end
ppfunc = getPreprocFunc(modtyp, subj);

[fnames1, mydir1] = subjtyp2dirs(subj, 'preictal', 'raw');
[fnames0, mydir0] = subjtyp2dirs(subj, 'interictal', 'raw');

% Check how many channels for this subject
Dat = loadSegFile(fullfile(mydir1,fnames1{1}));
nChn = size(Dat.data,1);

% Initialise
S1 = zeros(nChn,nChn);
C  = zeros(nChn,nChn);
lastSegID = 0;
lastSeqnc = 0;
nHour = 1;
% Load all the data
for iFle=1:length(fnames1)
    % Load this segment
    Dat = loadSegFile(fullfile(mydir1,fnames1{iFle}));
    % Apply the preprocessing function
    Dat = ppfunc(Dat);
    % Check the files are incremental!
    if Dat.segID ~= lastSegID+1
        error('Non-incremental segment IDs');
    end
    % Check if it is part of the same hour as the last one
    if Dat.sequence < lastSeqnc
        % New hour!
        nHour = nHour+1;
        % Add the covariance of the last hour to the total, normalising
        % against its variance
        S1 = S1 + C / trace(C);
        % Clear C to be zero again
        C  = zeros(nChn,nChn);
    end
    % Find the covariance estimate for segment and add to its hour's tally
    C = C + Dat.data * Dat.data';
    % Remember for next time
    lastSegID = Dat.segID;
    lastSeqnc = Dat.sequence;
end
% Add the final hour to the total
S1 = S1 + C / trace(C);
% Average the covariances from each hour
S1 = S1 / nHour;
% Display on screen
fprintf('Averaged over %d hours (%d segments) for preictal\n',nHour,length(fnames1));

% Initialise
S0 = zeros(nChn,nChn);
C  = zeros(nChn,nChn);
lastSegID = 0;
lastSeqnc = 0;
nHour = 1;
% Load all the data
for iFle=1:length(fnames0)
    % Load this segment
    Dat = loadSegFile(fullfile(mydir0,fnames0{iFle}));
    % Check the files are incremental!
    if Dat.segID ~= lastSegID+1
        error('Non-incremental segment IDs');
    end
    % Check if it is part of the same hour as the last one
    if Dat.sequence < lastSeqnc
        % New hour!
        nHour = nHour+1;
        % Add the covariance of the last hour to the total, normalising
        % against its variance
        S0 = S0 + C / trace(C);
        % Clear C to be zero again
        C  = zeros(nChn,nChn);
    end
    % Find the covariance estimate for segment and add to its hour's tally
    C = C + Dat.data * Dat.data';
    % Remember for next time
    lastSegID = Dat.segID;
    lastSeqnc = Dat.sequence;
end
% Add the final hour to the total
S0 = S0 + C / trace(C);
% Average the covariances from each hour
S0 = S0 / nHour;
% Display on screen
fprintf('Averaged over %d hours (%d segments) for interictal\n',nHour,length(fnames0));

% Compute CSP
W = csp(S1,S0);

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