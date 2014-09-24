% Correlation Coefficient feature
% Return a feature vector for each segment of certain type
function featM = feat_corrcoef(subj, typ)

% Get a list of files
[fnames, mydir, segIDs] = subjtyp2dirs(subj,typ);
nSeg = length(fnames);

% Check number of channels in this dataset
Dat = loadSegFile(fullfile(mydir,fnames{1}));
nChn = size(Dat.data,1);

% Initialise holding variable
featM = nan(nSeg,nChn^2);

% Loop over each segment, computing feature
for iSeg=1:nSeg
    % Load this segment
    Dat = loadSegFile(fullfile(mydir,fnames{iSeg}));
    % Compute covariance between channels
    featM(iSeg,:) = reshape(corrcoef(Dat.data'),1,[]);
end

end