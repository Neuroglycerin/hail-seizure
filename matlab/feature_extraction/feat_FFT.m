% First 250 datapoints of Fourier Transform of each channel
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 x nChn x 250 vector of FFT values
%        :[struct outparams]- structure with fields listing parameters used

function [featV,outparams] = feat_FFT(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'outlength', 250); % Number of points to use

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------
% Take fast-fourier transform
Dat.data = fft(Dat.data,[],2);

% Take first N values
featV = Dat.data(:, 1:param.outlength);

% Permute so correct shape for output
featV = permute(featV,[3 1 2]);

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;

end