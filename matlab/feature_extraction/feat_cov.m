% Cross-channel Covariance feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn*(nChn-1)/2) vector covariance of
%                             each pair of channels
%        :[struct outparams]- structure with fields listing parameters used
%
% NB: Excludes the variance terms. Use feat_var for single channel
% variance.

function [featV,outparams] = feat_cov(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('Covariance feature does not need any parameters. Dont provide any.');
end
% Empty params output
outparams = struct([]);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Compute the covariance
C = cov(Dat.data');

% Extract the upper triangle (excluding diagonal: use feat_var if you want
% the variance of each channel)
nPairs = nChn*(nChn-1)/2;

% Initialise holding variable
featV = nan(1,nPairs);

% Initialise pair counter
paircount = 0;
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        % Insert into feature vector
        featV(1,paircount) = C(iChn1,iChn2);
    end
end

end