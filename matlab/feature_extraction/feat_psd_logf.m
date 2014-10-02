% Power spectral density, sampled in logarithmically increasing bands
% Return a feature vector for each segment of certain type
% Inputs : str subj       - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str typ        - which datasegment type e.g. preictal, ictal, test
% Outputs: vec featm      - segment X (nChn*nBnd) matrix holding the PSD
%                           estimates in decibels
%        : vec bandEdges  - 2 X nBnd vector of frequencies used for bands
%                           1st row: lower bounds, 2nd row: upper bounds
%
% Note: the size of this feature will be different for different sampling frequencies

% To reshape so you can see the PSD for each channel individually (e.g. for
% plotting) run this command:
% featMM = permute(reshape(featM',size(bandEdges,2),[],size(featM,1)),[3 1 2]);
% figure; plot(mean(bandEdges,1), featMM(1,:,1));
% figure; plot(mean(bandEdges,1), featMM(iSeg,:,iChn));
% figure; boundedline(mean(bandEdges,1), mean((featMM(:,:,iChn))), std((featMM(:,:,iChn)))/sqrt(size(featMM,1)));


function [featM, bandEdges] = feat_psd_logf(subj, typ)

overlap = 0.5; % Proportional overlap to use
frqintv = 0.2; % Hz % Frequency interval
startwidth = 1; % Hz % Initial frequency band width
startf = 1; % Hz % Starting band frequency
base = 1.2; % Width multiplier

% Initial parameter set
% overlap = 0.5;
% frqintv = [];
% startwidth = 1;
% startf = 1;
% base = (1+sqrt(5))/2; % phi

% ------------------------------------------------------------------------
% Get segment files using io functions
[fnames, mydir, segIDs] = subjtyp2dirs(subj, typ);

% Number of segments of this subj and typ
nSeg = length(fnames);

% Check number of channels and num_samples in this dataset
% Assumes nChn and freq is constant within subj and typ
Dat = loadSegFile(fullfile(mydir, fnames{1}));
[nChn, data_points] = size(Dat.data);

% ------------------------------------------------------------------------

if isempty(frqintv); frqintv = startwidth; end;

% Window length in seconds should be reciprocal of frequency interval
window_dur = 1/frqintv;

% Number of datapoints in window
window_size = window_dur*Dat.fs;

% Round to nearest even number
window_size = round(window_size/2)*2;

% Use hanning windowing
wndw = hanning(window_size);

% Length of FFT to use should match the length of the window
Nfft = window_size;

% Work out how many frequencies we will get out
nFrq = floor(Nfft/2)+1;


% Work out spacing between frequency samples
frqintv = 1/(window_size/Dat.fs);

% Round band width to be a multiple of the frequency interval
% express in terms of number of datapoints to include
startwidthIdx = max(1,round(startwidth/frqintv));
startwidth    = startwidthIdx*frqintv;

% Work out the vector of frequencies
f = (0:nFrq-1)*frqintv;

% Find out how many bands of the initial size we can fit in before we start
% making the bands bigger
n_pre = ceil(startf/startwidth);

% Indices at which each of the base bands begin
x_pre = (0:n_pre-1)*startwidthIdx;

% Work out index where we have the last band before doubling
x0 = n_pre*startwidthIdx;
% Number of datapoints used doubles every time - logarithmic spacing
pwrmax = log(f(end)/startwidth)/log(base);
x_post = round(startwidthIdx * base.^(0:pwrmax)); % Index width of each band
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
featM = nan(nSeg,nChn*nBnd);

% ------------------------------------------------------------------------
% Loop over each segment, computing feature
for iSeg=1:nSeg
    
    % Load this segment
    Dat = loadSegFile(fullfile(mydir, fnames{iSeg}));
    
    % Iterate over each channel
    for iChn=1:nChn
        % Use pwelch to calculate spectra and frequency for each channel
        [Pxx, f] = pwelch(Dat.data(iChn,:), ...
            wndw, overlap, Nfft, Dat.fs);
        % Sum over each band
        for iBnd=1:nBnd
            bndpwr = sum(Pxx(x(iBnd):x(iBnd+1)-1));
            featM(iSeg,(iChn-1)*nBnd+iBnd) = 10*log10(bndpwr);
        end
    end
    bandEdges = nan(2,nBnd);
    for iBnd=1:nBnd
        bandEdges(:,iBnd) = f([x(iBnd) x(iBnd+1)-1]);
    end
end

end