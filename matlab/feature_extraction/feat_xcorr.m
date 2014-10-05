% Cross-correlation feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) X 2 vector holding the
%                             peak cross-correlation amount and lag
%                             for each unique channel pairing
%        :[struct outparams]- structure with fields listing parameters used

function [featV, outparams] = feat_xcorr(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'maxlagdur'      , 2         , ... % maximum lag duration in seconds
    'band'           , [0 Inf]   , ... % Filtering band (default: none)
    'signalattribute', 'none'    );    % Signal processing

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% ------------------------------------------------------------------------

% Check number of channels and num_samples in this dataset
nChn = size(Dat.data,1);

% ------------------------------------------------------------------------

maxlagidx = ceil(param.maxlagdur * Dat.fs);

% Initialise holding variable
featV = nan(1, nChn^2, 2);

% Permute data so it is [nPnt x nChn]
Dat.data = Dat.data';

% Filter to band
Dat.data = butterfiltfilt(Dat.data, param.band, Dat.fs);

switch param.signalattribute
    case 'none'
        % Do nothing
    case 'envelope'
        % Find envelope amplitude
        Dat.data = abs(hilbert(Dat.data));
    case 'power'
        % Find envelope amplitude
        Dat.data = abs(hilbert(Dat.data)).^2;
    otherwise
        error('Unfamiliar signal processing attribute: %s',param.signalattribute);
end

% Compute cross-correlation between channels
C = xcorr(Dat.data, maxlagidx, 'unbiased');

% Find minima and maxima
[~,Imax] = max(abs(C));

% Vector of the lag of each datapoint in columns of C
lagvec = (-maxlagidx:maxlagidx) / Dat.fs;

% Take out the min/max cross-correlation and its lag
featV(1,:,1) = diag(C(Imax,:));
featV(1,:,2) = lagvec(Imax);

% Reduce to only interesting parts of this
% Not interested in iChn1xiChn2 AND iChn2xiChn1, or in iChn1xiChn1
intidx = false(1,nChn*(nChn-1)/2);
for iChn1=1:nChn
    intidx((iChn1-1)*nChn+((iChn1+1):nChn)) = true;
end
featV = featV(1,intidx,:);

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;

end