% Computes the mutual information between each element of the feature
% vector and the label of the ictal period

function [I,Ierr] = computeFeatInfo(feat0, feat1)

% Declarations ------------------------------------------------------------
% Information options:
InfoOpt = struct(...
    'method'  , 'dr', ...
    'bias'    , 'pt', ...
    'btsp'    , 20  , ...
    'binf'    , 'eqpop');

% Input handling ----------------------------------------------------------
if iscell(feat0)
    feat0 = handleCellInput(feat0);
end
if iscell(feat1)
    feat1 = handleCellInput(feat1);
end

% Main --------------------------------------------------------------------
% Check sizing of features
siz0 = size(feat0);
siz1 = size(feat1);
if ~isequal(siz0(2:end),siz1(2:end)); error('unbalanced sizes'); end;
siz  = [1 siz0(2:end)];

% Number of trials for each ictal type
InfoOpt.nt = [siz0(1); siz1(1)];
% Check this is a plausible number of trials to compute information on
if any(InfoOpt.nt<8);
    error('To few trials to get an information estimate');
end

% Initialise output variables
I    = nan(siz);
Ierr = nan(siz);

Rsiz = [1 max(siz0(1),siz1(1)) 2];

% Loop over every element in the feature vector
% Could parallelise, but its quick so not much need
for iElm = 1:prod(siz(2:end))
    
    % Merge together the two ictal types for this element of the feature
    % vector --------------------------------------------------------------
    R_raw = nan(Rsiz);
    R_raw(1, 1:InfoOpt.nt(1), 1) = feat0(:,iElm);
    R_raw(1, 1:InfoOpt.nt(2), 2) = feat1(:,iElm);
    
    % Bin the responses with a histogram ----------------------------------
    % Check the number of possible responses for this element of the
    % feature vector
    nUnq = length(unique(R_raw(~isnan(R_raw))));
    % If this element is always the same, it contains no information
    if nUnq<2
        I(1,iElm) = 0;
        Ierr(1,iElm) = 0;
        continue;
    end
    % The number of bins to use should be no more than a quarter of the
    % least populous experimental paradigm to ensure accuracy
    nBin = floor(min(InfoOpt.nt)/4);
    % However, we cannot have more bins than possible responses
    nBin = min(nUnq, nBin);
    
    % Perform the binning
    R_bin = binr(R_raw, InfoOpt.nt, nBin, InfoOpt.binf);
    
    % Compute information -------------------------------------------------
    myI = information(R_bin, InfoOpt, 'I');
    % Correct for residual bias by subtracting average of bootstraps
    I(1,iElm) = myI(1) - mean(myI(2:end));
    % Error on the information is approximately the standard deviation of
    % the bootstraps
    Ierr(1,iElm) = std(myI(2:end));
    
end

end

% Helper fuction to convert featM in cell style into a matrix
function M = handleCellInput(C)

% Check cell is how we expect it to be
if ndims(C)>2; error('Input has too many dims'); end
if size(C,2)~=2; error('Input is badly shaped'); end;
if ~ischar(C{1,1}); error('First cell column should be strings'); end;
if ~isnumeric(C{1,2}); error('Second cell column should be numeric'); end;
% Assume all the other rows are okay if the first one is

% Number of segment files is the number of rows of cell
nSeg = size(C,1);

% Get size of first feature vector
siz = size(C{1,2});
if siz(1)~=1; error('Cannot merge files along first dimension'); end;

% Initialise holding matrix
siz(1) = nSeg;
M = nan(siz);
% Add each feature vector into the matrix
for iSeg=1:nSeg
    M(iSeg,:) = C{iSeg,2};
end

end