% Power in band feature, computed via power spectral density, relative to
% the amount of broadband power
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x numchannels x numbands vector holding the
%                             power-in-band values
%        :[struct outparams]- structure with fields listing parameters used

function [featV, outparams] = feat_pib_ratioBB(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------

% Bands used in Howbert et al., 2014
defaultbands = struct(...
    'delta',      [ 0.1    4.0],  ...
    'theta',      [ 4.0    8.0],  ...
    'alpha',      [ 8.0   12.0],  ...
    'beta',       [12.0   30.0],  ...
    'low_gamma',  [30.0   70.0],  ...
    'high_gamma', [70.0  180.0],  ...
    'broadband',  [ 0.1  180.0]); % Measure broadband power last

defparams = struct(...
    'overlap', 0.5 ,...
    'bands'  , defaultbands,...
    'window' , []);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% ------------------------------------------------------------------------
% Compute power in each band
[featV, outparams] = feat_pib(Dat, param);

% Check size is as we expect
if ndims(featV)~=3
    error('Size mismatch');
end
if size(featV,3)~=length(fieldnames(param.bands))
    error('Band count mismatch');
end

% Divide by broadband power
featV = bsxfun(@rdivide, featV(1,:,1:end-1), featV(1,:,end));

end
