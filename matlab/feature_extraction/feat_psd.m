% Power spectral density feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn X nFrq vector holding the PSD
%                             estimates in decibels
%                             where nFrq is round(1+samplingfreq/2)
%        :[struct outparams]- structure with fields listing parameters used
%
% Note: the size of this feature will be different for different sampling frequencies

% To reshape so you can see the PSD for each channel individually (e.g. for
% plotting) run this command:
% featV = permute(reshape(featV',length(outparams.f),[],size(featV,1)),[3 1 2]);
% figure; plot(outparams.f, featV(1,:,iChn));

function [featV, outparams] = feat_psd(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'overlap', 0.5 ,...
    'frqintv', 1   );

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% ------------------------------------------------------------------------

% Check number of channels and num_samples in this dataset
nChn = size(Dat.data,1);

% ------------------------------------------------------------------------

% Window length in seconds should be reciprocal of frequency interval
window_dur = 1/param.frqintv;

% Number of datapoints in window
wndw_size = window_dur*Dat.fs;

% Round to nearest even number
wndw_size = round(wndw_size/2)*2;

% Use hanning windowing
wndw = hanning(wndw_size);

% Length of FFT to use should match the length of the window
Nfft = wndw_size;

% Work out how many frequencies we will get out
nFrq = floor(Nfft/2)+1;

% Initialise holding variable
featV = nan(1,nChn,nFrq);

% ------------------------------------------------------------------------
% Iterate over each iChn
for iChn=1:nChn
    % Use pwelch to calculate spectra and frequency for each channel
    [Pxx, f] = pwelch(Dat.data(iChn,:), ...
        wndw, param.overlap, Nfft, Dat.fs);
    % Convert to dB and add PSD to feature matrix
    featV(1,iChn,:) = 10*log10(Pxx);
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.window_dur = window_dur;
outparams.window     = wndw;
outparams.nfft       = Nfft;
outparams.f          = f;

end