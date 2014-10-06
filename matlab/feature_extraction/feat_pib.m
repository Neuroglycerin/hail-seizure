% Power in band feature, computed via power spectral density
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X numbands X numchannels vector holding the
%                             power-in-band values
%        :[struct outparams]- structure with fields listing parameters used

function [featV, outparams] = feat_pib(Dat, inparams)

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
    'high_gamma', [70.0  180.0]);

defparams = struct(...
    'overlap', 0.5 ,...
    'bands'  , defaultbands,...
    'window' , []);

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

% ------------------------------------------------------------------------

% Check number of channels and num_samples in this dataset
[nChn, num_samples] = size(Dat.data);

% ------------------------------------------------------------------------

% Window size (hanning) for pwelch unless specified otherwise
% Sqrt of number of samples rounded up to nearest int
if isempty(param.window)
    wndw_size  = floor(sqrt(num_samples));
    param.window = hanning(wndw_size);
end

% Check what the interval between frequency samples will be
% Total duration of windowed segement
window_dur = wndw_size/Dat.fs;
% Frequencies are (1:N) times per window, so in Hz we have
frqintv = 1/window_dur;

% Length of FFT to use
Nfft = wndw_size;

% Check what bands we have
bandnames = fieldnames(param.bands);
nBnd = length(bandnames);

% Initialise holding variable
featV = nan(1,nChn,nBnd);
bandsused = struct([]);

% Iterating over each channel
for iChn=1:nChn
    
    % Use pwelch to calculate spectra and frequency for each channel
    [Pxx, f] = pwelch(Dat.data(iChn,:), ...
        param.window, param.overlap, Nfft, Dat.fs);
    
    % For 
    % Integrate frequency bands on spectra and append to output
    % matrix of Power-In-Band values
    for iBnd = 1:nBnd
        bandfreq = param.bands.(bandnames{iBnd});
        idx = (f>=(bandfreq(1)-frqintv/2) & f<=(bandfreq(2)+frqintv/2));
        bandpwr = trapz(f(idx), Pxx(idx));
        featV(1,iChn,iBnd) = bandpwr;
        bandsused(1).(bandnames{iBnd}) = f([find(idx,1,'first'),find(idx,1,'last')]);
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.window_dur = window_dur;
outparams.frqintv    = frqintv;
outparams.nfft       = Nfft;
outparams.f          = f;

end
