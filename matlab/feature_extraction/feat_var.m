% Single channel Variance feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X (nChn) vector variance of each channel
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_var(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% No paramters needed
if ~isempty(inparams)
    error('Variance feature does not need any parameters. Dont provide any.');
end
% Empty params output
outparams = struct([]);

% Main --------------------------------------------------------------------
% Compute the variance
featV = var(Dat.data,[],2);

% Turn into a row vector
featV = featV(:)';

end