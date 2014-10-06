% Cross-channel spectral coherence feature
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn*(nChn-1)/2 X nFrq vector holding the
%                             coherence values
%                             where nFrq is round(1+samplingfreq/2)
%        :[struct outparams]- structure with fields listing parameters used
%
% Note: the size of this feature will be different for different sampling frequencies

% To reshape so you can see the coherence for each individually pair of
% channels (e.g. for plotting) run this command:
% featMM = permute(reshape(featV',length(outparams.f),[],size(featV,1)),[3 1 2]);
% figure; plot(outparams.f, featMM(1,:,1));
% figure; plot(outparams.f, featMM(iSeg,:,iPair));

function [featV, outparams] = feat_coher(Dat, inparams)

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

% Work out how many frequencies we will get out for each channel pair
nFrq = floor(Nfft/2)+1;

% Initialise holding variable
featV = nan(1, nChn*(nChn-1)/2, nFrq);

% ------------------------------------------------------------------------

% Initialise pair counter
paircount = 0;

% Iterate over each pair of channels
% Only do each pair of channels once (symmetric measure)
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        
        % Use pwelch to calculate cross-channel coherence
        [coher, f] = pwelch(Dat.data(iChn1,:), Dat.data(iChn2,:),...
            wndw, param.overlap, Nfft, Dat.fs, 'coher');
        
        % Set correct part of feature row
        featV(1,paircount,:) = coher;
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.window_dur = window_dur;
outparams.window     = wndw;
outparams.nfft       = Nfft;
outparams.f          = f;

end