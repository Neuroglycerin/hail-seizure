% Power in band feature, computed via power spectral density, taking ratio
% of power in each band
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x nChn x nBnd*(nBnd-1)/2 vector holding the
%                             power-in-band ratio values
%        :[struct outparams]- structure with fields listing parameters used

function [featV, outparams] = feat_pib_ratio(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
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
    'overlap', 0.5 ,...
    'bands'  , defaultbands,...
    'frqintv', 1   );

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% ------------------------------------------------------------------------
% Compute power in each band
[featP, outparams] = feat_pib(Dat, param);

% Check size is as we expect
if ndims(featP)~=3
    error('Size mismatch');
end
if size(featP,3)~=length(fieldnames(param.bands))
    error('Band count mismatch');
end

[nPrt,nChn,nBnd] = size(featP);

% Use upper triangle only
nPairs = nBnd*(nBnd-1)/2;

% Initialise holding variable
featV = nan(nPrt, nChn, nPairs);

% Initialise pair counter
paircount = 0;
for iBnd1=1:nBnd
    for iBnd2=(iBnd1+1):nBnd
        % Track how many pairs we have done
        paircount = paircount + 1;
        % Take ratio between these bands
        featV(:,:,paircount) = featP(:,:,iBnd1) ./ featP(:,:,iBnd2);
    end
end

end
