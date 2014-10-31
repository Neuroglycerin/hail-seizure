% Single-channel L-moments
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X momentorder vector L-moments
%                             of orders 1, ..., momentorder for
%                             each channel. Orders above 2 are standardised
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_lmom(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% Empty params output
outparams = struct([]);

% Default parameters ------------------------------------------------------

defparams = struct(...
    'maxorder', 6);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Initialise holding variable
featV = nan(1, nChn, param.maxorder);

for iChn=1:nChn
    L = lmom(Dat.data(iChn,:)',param.maxorder);
    featV(1,iChn,:) = L;
    if length(L)>2
        featV(1,iChn,3:end) = L(3:end)/L(2);
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.featnames = cell(1,1,param.maxorder);
for iOrd=1:param.maxorder
    outparams.featnames{iOrd} = num2str(iOrd);
end

end