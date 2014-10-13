% Cross-channel Correlation coefficient feature of PSD
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x (nChn*(nChn-1)/2) vector correlation
%                             coefficient of each non-trivial pair of channels
%                             after taking PSD
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_PSDlogfcorrcoefeig(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% Default parameters ------------------------------------------------------
defparams = struct([]);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Take psd with logarithmic sampling
[featPSD, outparams] = feat_psd_logf(Dat, param);

% Merge in these parameters
param = parammerge(param, outparams, 'union');

% Permute so channels are in dimension 2 again
Dat.data = permute(featPSD,[2 3 1]);

% Pass to corrcoef function
[featV,outparams] = feat_corrcoefeig(Dat, inparams);

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = parammerge(param, outparams, 'union');

end