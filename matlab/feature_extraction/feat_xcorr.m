% Cross-correlation feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x (nChn*(nChn-1)/2) x 3 vector holding:
%                             1 x (nChn*(nChn-1)/2) x[1]... peak xcorr value
%                             1 x (nChn*(nChn-1)/2) x[2]... peak xcorr lag
%                             1 x (nChn*(nChn-1)/2) x[3]... peak xcorr width
%                             for each unique channel pairing
%        :[struct outparams]- structure with fields listing parameters used

function [featV, outparams] = feat_xcorr(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'maxlagdur'      , 3         , ... % maximum lag duration in seconds
    'band'           , []        , ... % Filtering band (default: below)
    'signalattribute', 'none'    , ... % Signal processing
    'scaleopt'       , 'coeff'   , ... % How to scale cross-correlation
    'widthmag'       , exp(-1)   );    % 

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% By default, eliminate slow frequencies below the duration we are
% considering
if isempty(param.band)
    param.band = [1/param.maxlagdur Inf];
end

% ------------------------------------------------------------------------

% Check number of channels and num_samples in this dataset
nChn = size(Dat.data,1);

% ------------------------------------------------------------------------

maxlagidx = ceil(param.maxlagdur * Dat.fs);

% Vector of the lag of each datapoint in columns of C
lagvec = (-maxlagidx:maxlagidx) / Dat.fs;

% Permute data so it is [nPnt x nChn] for filtering
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

% Permute data back again so it is [nChn x nPnt]
Dat.data = Dat.data';

% Don't compute all cross-correlations at once because it uses too much
% memory. Instead take each pair one at a time.
featV = nan(1, nChn*(nChn-1)/2, 3);

% Initialise pair counter
paircount = 0;

for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        
        % Compute cross-correlation between channels
        C = xcorr(Dat.data(iChn1,:), Dat.data(iChn2,:), maxlagidx, param.scaleopt);
        
        % Find minima or maxima
        [~,Imax] = max(abs(C));
        Cmax = C(Imax);
        
        % Take out the min/max cross-correlation and its lag
        featV(1,paircount,1) = Cmax;
        featV(1,paircount,2) = lagvec(Imax);
        
        % Find the width of the peak
        if Cmax>0
            % Positive cross-correlation
            ind = find(C <= Cmax * param.widthmag);
        else
            % Negative cross-correlation
            ind = find(C >= Cmax * param.widthmag);
        end
        % Find left and right distances in sampling indices from peak
        ind = ind - Imax;
        lft = ind(find(ind<0,1,'last'));
        rgt = ind(find(ind>0,1,'first'));
        % Sum left and right distances from peak
        wdth = (abs(lft)+abs(rgt)) / Dat.fs;
        
        featV(1,paircount,3) = wdth;
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;

end
