% Cross-channel phase synchrony
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) X 2*nBnd vector 
%                             in dim 2, all channel pairings
%                             in dim 3, band synchrony followed by mean
%                             phase difference for each band
%        :[struct outparams]- structure with fields listing parameters used
%
% NB: Excludes the variance terms. Use feat_var for single channel
% variance.

function [featV,outparams] = feat_phase(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------

% Bands used in Howbert et al., 2014
defaultbands = struct(...
    'delta',      [ 1.0    4.0],  ...
    'theta',      [ 4.0    8.0],  ...
    'alpha',      [ 8.0   12.0],  ...
    'beta',       [12.0   30.0],  ...
    'low_gamma',  [30.0   70.0],  ...
    'high_gamma', [70.0  180.0]);

defparams = struct(...
    'order'  , 2   ,... % Filter order
    'bands'  , defaultbands);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Check what bands we have
bandnames = fieldnames(param.bands);
nBnd = length(bandnames);

% Fill use the upper triangle (excluding diagonal)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = nan(1,nPairs,nBnd*2);
param.featnames = cell(1,1,nBnd*2);

% Loop over each band
for iBnd=1:nBnd
    % Filter within this band
    fltDat = butterfiltfilt(Dat.data', param.bands.(bandnames{iBnd}), Dat.fs, param.order)';
    % Take phase with Hilbert transform
    fltDat = angle(hilbert(fltDat'))';
    % Initialise pair counter
    paircount = 0;
    for iChn1=1:nChn
        for iChn2=(iChn1+1):nChn
            % Track how many pairs we have done
            paircount = paircount + 1;
            % Take the difference between the phases
            phsDif = fltDat(iChn1,:) - fltDat(iChn2,:);
            % Take the vector mean of the phase differences
            r = sum(exp(1i*phsDif))/length(phsDif);
            % Insert into feature vector
            featV(1,paircount,iBnd*2-1) = abs(r);
            featV(1,paircount,iBnd*2  ) = angle(r);
        end
    end
    param.featnames{1,1,iBnd*2-1} = [bandnames{iBnd} '-sync'];
    param.featnames{1,1,iBnd*2  } = [bandnames{iBnd} '-dif'];
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;

end