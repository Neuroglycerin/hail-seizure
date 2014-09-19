% Covariance feature
% Return a feature vector for each segment of certain type
function featM = feat_cov(subj,typ)

% Get a list of files
[myfiles,mydir] = subjtyp2dirs(subj,typ);
nSeg = length(myfiles);

% Check number of channels in this dataset
Dat = loadSegFile(fullfile(mydir,myfiles(1).name));
nChn = size(Dat.data,1);

% Initialise holding variable
featM = nan(nSeg,nChn^2);

% Loop over each segment, computing feature
for iSeg=1:nSeg
    % Load this segment
    Dat = loadSegFile(fullfile(mydir,myfiles(iSeg).name));
    % Compute covariance between channels
    featM(iSeg,:) = reshape(cov(Dat.data'),1,[]);
end

end