% Cross-channel Correlation coefficient Eigenvalues feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn vector of eigenvalues for correlation
%                             coefficient matrix
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_corrcoefeig(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('Correlation coefficient eigenvalue feature does not need any parameters. Dont provide any.');
end
% Empty params output
outparams = struct([]);

% Main --------------------------------------------------------------------
% Check number of channels in this dataset
nChn = size(Dat.data,1);

% Compute the covariance
C = corrcoef(Dat.data');

% Initialise output so it is right shape
featV = nan(1,nChn);

% Compute eigenvalues
featV(1,:) = eig(C);

end