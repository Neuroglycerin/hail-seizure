% Cross-channel Correlation coefficient Eigenvalues of Fourier Transform
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x nChn vector of eigenvalues for correlation
%                             coefficient matrix on FFT of data
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_FFTcorrcoefeig(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2;
    inparams = struct([]);
end
% Default parameters ------------------------------------------------------
defparams = struct(...
    'slice'  , [1 250] );

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Take fast-fourier transform
Dat.data = fft(Dat.data,[],2);

% Take first N values
Dat.data = Dat.data(:, param.slice(1):param.slice(2));

% Take absolute value
Dat.data = abs(Dat.data);

% Pass to corrcoefeig function
[featV,outparams] = feat_corrcoefeig(Dat, param);

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = parammerge(param, outparams, 'union');

end