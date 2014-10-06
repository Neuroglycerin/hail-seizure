% Coherence, sampled in logarithmically increasing bands
% Inputs : struct Dat       - structure loaded from segment file
%        :[struct inparams] - structure with fields listing requested parameters
%
% Outputs: vec featV        - feature vector
%                             1 X nChn*(nChn-1)/2 X nBnd vector holding the 
%                             coherence values
%        :[struct outparams]- structure with fields listing parameters used
%
% Note: the size of this feature will be different for different sampling frequencies

% To reshape so you can see the PSD for each channel individually (e.g. for
% plotting) run this command:
% featMM = permute(reshape(featV',size(outparams.bandEdges,2),[],size(featV,1)),[3 1 2]);
% figure; plot(mean(outparams.bandEdges,1), featMM(1,:,1));
% figure; plot(mean(outparams.bandEdges,1), featMM(iSeg,:,iChn));
% figure; boundedline(mean(outparams.bandEdges,1), mean((featMM(:,:,iChn))), std((featMM(:,:,iChn)))/sqrt(size(featMM,1)));

function [featV, outparams] = feat_coher_logf(Dat, inparams)

% Default inputs ----------------------------------------------------------
if nargin<2
    inparams = struct([]);
end

% Default parameters ------------------------------------------------------
defparams = struct(...
    'overlap'   , 0.5 ,...
    'frqintv'   , 0.2 ,...
    'startwidth', 1   ,...
    'startf'    , 1   ,...
    'base'      , 1.2 );

% Overwrite default parameters with input parameters
param = parammerge(defparams, inparams);

if isempty(param.frqintv); param.frqintv = param.startwidth; end;

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


% Work out spacing between frequency samples
param.frqintv = 1/(wndw_size/Dat.fs);

% Round band width to be a multiple of the frequency interval
% express in terms of number of datapoints to include
startwidthIdx = max(1,round(param.startwidth/param.frqintv));
param.startwidth    = startwidthIdx*param.frqintv;

% Work out the vector of frequencies
f = (0:nFrq-1)*param.frqintv;

% Find out how many bands of the initial size we can fit in before we start
% making the bands bigger
n_pre = ceil(param.startf/param.startwidth);

% Indices at which each of the param.base bands begin
x_pre = (0:n_pre-1)*startwidthIdx;

% Work out index where we have the last band before doubling
x0 = n_pre*startwidthIdx;
% Number of datapoints used doubles every time - logarithmic spacing
pwrmax = log(f(end)/param.startwidth)/log(param.base);
x_post = round(startwidthIdx * param.base.^(0:pwrmax)); % Index width of each band
x_post = cumsum(x_post); % Cumsum to find start indices
% Shift so we start at correct frequency index
x_post = x_post + x0;

% Merge before and after doubling indices for full list
x = 1+[0 x_pre x0 x_post];
x = unique(x);
% Cut off the indices which are too big
x = x(x<(nFrq-startwidthIdx));
% Final band runs from where we stop to the end
x(end+1) = length(f);

% Number of logarithmic bands
nBnd = length(x)-1;

% Initialise holding variable
featV = nan(1, nChn*(nChn-1)/2, nBnd);

% ------------------------------------------------------------------------

% Store each band power
bndpwr = nan(nChn,nBnd);

% Iterate over each channel
for iChn=1:nChn
    % Use pwelch to calculate spectra and frequency for each channel
    [Pxx, f] = pwelch(Dat.data(iChn,:), ...
        wndw, param.overlap, Nfft, Dat.fs);
    % Sum over each band
    for iBnd=1:nBnd
        bndpwr(iChn,iBnd) = sum(Pxx(x(iBnd):x(iBnd+1)-1));
    end
end
clear iChn;

% Note frequencies of band edges
bandEdges = nan(2,nBnd);
for iBnd=1:nBnd
    bandEdges(:,iBnd) = f([x(iBnd) x(iBnd+1)-1]);
end

% Initialise pair counter
paircount = 0;

% Iterate over each channel
for iChn1=1:nChn
    for iChn2=(iChn1+1):nChn
        % Track how many pairs we have done
        paircount = paircount + 1;
        
        % Use pwelch to calculate spectra and frequency for each channel
        Pxy = pwelch(Dat.data(iChn1,:), Dat.data(iChn2,:), ...
            wndw, param.overlap, Nfft, Dat.fs, 'cross');
        crspwr = nan(1,nBnd);
        
        % Sum over each band
        for iBnd=1:nBnd
            crspwr(1,iBnd) = sum(abs(Pxy(x(iBnd):x(iBnd+1)-1)));
        end
        
        featV(1,paircount,:) = abs(crspwr).^2 ./ (bndpwr(iChn1,:).*bndpwr(iChn2,:));
    end
end

% ------------------------------------------------------------------------
% Determine output parameter structure
outparams = param;
outparams.window_dur = window_dur;
outparams.window     = wndw;
outparams.nfft       = Nfft;
outparams.bandEdges  = bandEdges;

end