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
    
    % Get segment files using io functions
    [fnames, mydir, segIDs] = subjtyp2dirs(subj, typ);
    
    % Number of segments of this subj and typ
    nSeg = length(fnames);

    % Check number of channels and num_samples in this dataset
    % Assumes nChn and freq is constant within subj and typ
    Dat = loadSegFile(fullfile(mydir, fnames{1}));
    [nChn, data_points] = size(Dat.data);

    % Window size (hanning) for pwelch unless specified otherwise
    % Sqrt of number of samples rounded up to nearest int
    if nargin < 3
           window_size = ceil(sqrt(length(Dat.data(1,:))));
           window = hanning(window_size);
    end

    % Pwelch window to use no overlap setting
    overlap = 0;

    % Length of FFT to use
    Nfft = window_size;

    
    % Initialise holding variable
    featM = []; 

    % Loop over each segment, computing feature
    for iSeg=1:nSeg
        
        % Load this segment
        Dat = loadSegFile(fullfile(mydir, fnames{iSeg}));
        
        % Initialise PIB holding variable
        segment_PIBs = [];
        
        % Iterating over each channel
        for channel=1:nChn
            
            % Use pwelch to calculate spectra and frequency for each
            % channel
            [Pxx, f] = pwelch(Dat.data(channel,:), ...
                              window, ...
                              overlap, ...
                              Nfft, ...
                              Dat.fs);
             
             % Integrate frequency bands on spectra and append to output
             % matrix of Power-In-Band values
             for i = fieldnames(bins)'
                freq_bin = bins.(i{1});
                idx = find(f>=freq_bin(1) & f<=freq_bin(2));
                density = trapz(f(idx), Pxx(idx));
                segment_PIBs = [segment_PIBs, density];
             end
        end
        % Append PIBs to feature matrix so each segment has a separate PIB
        % row
        featM = [featM; segment_PIBs];
    end
end