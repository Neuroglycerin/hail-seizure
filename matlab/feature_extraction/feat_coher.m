% Cross-channel spectral coherence feature
% Return a feature vector for each segment of certain type
% Inputs : str subj       - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str typ        - which datasegment type e.g. preictal, ictal, test
%        : [float frqintv]- Interval of frequencies in the PSD
% Outputs: vec featm      - segment X (nChn*(nChn-1)/2*nFrq) matrix holding the coherence values
%                           for each segment
%                           where nFrq is round(1+samplingfreq/2)
%        : vec f          - vector of frequencies 
%
% Note: the size of this feature will be different for different sampling frequencies 

% To reshape so you can see the coherence for each individually pair of
% channels (e.g. for plotting) run this command:
% featMM = permute(reshape(featM',length(f),[],size(featM,1)),[3 1 2]);
% figure; plot(f, featMM(1,:,1));
% figure; plot(f, featMM(iSeg,:,iPair));

function [featM, f] = feat_coher(subj, typ, frqintv)

% Default frequency interval
if nargin<3;
    frqintv = 1; %Hz
end
% Proportional overlap to use
overlap = 0.5;

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

% Window length in seconds should be reciprocal of frequency interval
window_dur = 1/frqintv;

% Number of datapoints in window
window_size = window_dur*Dat.fs;

% Round to nearest even number
window_size = round(window_size/2)*2;

% Use hanning windowing
window = hanning(window_size);

% Length of FFT to use should match the length of the window
Nfft = window_size;

% Work out how many frequencies we will get out for each channel pair
nFrq = floor(Nfft/2)+1;

% Initialise holding variable
featM = nan(nSeg, nChn*(nChn-1)/2*nFrq);

% ------------------------------------------------------------------------
% Loop over each segment, computing feature
for iSeg=1:nSeg
    
    % Load this segment
    Dat = loadSegFile(fullfile(mydir, fnames{iSeg}));
    
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
                window, overlap, Nfft, Dat.fs, 'coher');
            
            % Set correct part of feature row
            featM(iSeg,(paircount-1)*nFrq+(1:nFrq)) = coher;
        end
    end
end

end