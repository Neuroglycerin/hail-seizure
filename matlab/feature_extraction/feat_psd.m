% Power spectral density feature
% Return a feature vector for each segment of certain type
% Inputs: str subj       - which subject e.g. Dog_[1-5], Patient_[1-2]
%       : str typ        - which datasegment type e.g. preictal, ictal, test
%       : [float window] - window length (seconds) to use within segment 
%                        (need add code handling the frequency of that seg)                        
%       : [struct bins]  - floats of the freq bins to use (Hz)
% Output: vec featm      - floats of 

function featM = feat_psd(subj, typ, window, bins)
    
    % Hack to have default args in matlab
    % Default args were those used in Howbert et al., 2014
    if nargin < 4
        bins = struct('delta',      [ 0.1    4.0],  ...
                      'theta',      [ 4.0    8.0],  ...
                      'alpha',      [ 8.0   12.0],  ...
                      'beta',       [12.0   30.0],  ...
                      'low_gamma',  [30.0   70.0],  ...
                      'high_gamma', [70.0  180.0]); 
    end
    
    if nargin < 3
        window = 60;  
    end
    
% Get segment files using io functions
[fnames, mydir, segIDs] = subjtyp2dirs(subj, typ);

nSeg = length(fnames);

% Check number of channels and num_samples in this dataset
% Assumes nChn and freq is constant within subj and typ
Dat = loadSegFile(fullfile(mydir, fnames{1}));
[nChn, data_points] = size(Dat.data);

% Num of rows per window (frequency * window size)
subsection_length = int32(Dat.sampling_frequency * window); 

% Within each frequency
% band the power was summed over band frequencies to produce a
% power-in-band (PIB) feature. These features were aggregated into a
% feature vector containing 96 PIB values 

% Initialise holding variable
featM = nan(nSeg, nChn*length(fieldnames(bins)));

% Loop over each segment, computing feature
for iSeg=1:nSeg
    % Load this segment
    Dat = loadSegFile(fullfile(mydir, fnames{iSeg}));
    
    Nfft = [];
    overlap = 0;
   
    
    for channel=1:nChn
        
        [Pxx, f] = pwelch(Dat.data(channel,:), 600, overlap, Nfft, Dat.fs);

        % Bin the spectral densities according to bins
    
    % Return spectral densities
     %[spectra,freq] = pwelch(x,window,overlap,Nfft,Fs,
                            %range,plot_type,detrend,sloppy)%
       
%   The power of the signal in a given frequency band [\omega_1,\omega_2] can be calculated by integrating over positive and negative frequencies,
% Use trapezoidal numerical integration to approximate power in each band
% trapz(X,Y,DIM) integrates across dimension DIM
 %   of Y. The length of X must be the same as size(Y,DIM))
%  pwelch() to get the spectrum and trapz() to
    % featM(iSeg,:) = reshape(cov(Dat.data'),1,[]);
end

end