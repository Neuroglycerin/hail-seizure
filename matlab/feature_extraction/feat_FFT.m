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
    'slice', [1 250]); % Number of points to use

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% Main --------------------------------------------------------------------

% Check number of channels and num_samples in this dataset
nChn = size(Dat.data,1);

% Take fast-fourier transform
Dat.data = fft(Dat.data,[],2);

% Take first N values
Dat.data = Dat.data(:, param.slice(1):param.slice(2));

% Initialise output
featV = nan(1, nChn, diff(param.slice)+1, 2);

% Store real and imaginary parts separately
featV(1,:,:,1) = real(Dat.data);
featV(1,:,:,2) = imag(Dat.data);

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.window_dur = size(Dat.data,2)/Dat.fs;
outparams.f          = (param.slice(1):param.slice(2))/outparams.window_dur;

end