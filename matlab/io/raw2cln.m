function Dat = raw2cln(Dat)

% Parameters
order = 2;

% No need to clean if sampling rate is low
if Dat.sampling_frequency<600
    return;
end

% Note: apply the same preprocessing model to every dataset

% Remove 60Hz line noise artifact (present in Patient_1)
% Notch reject
Dat.data = butterfiltfilt(Dat.data', [55 65], Dat.sampling_frequency, order, 'stop', 'both')';

% Remove 180Hz line noise artifact harmonic (present in Patient_1)
Dat.data = butterfiltfilt(Dat.data', [175 185], Dat.sampling_frequency, order, 'stop', 'both')';

% Remove 240Hz line noise artifact harmonic (present in Patient_1)
Dat.data = butterfiltfilt(Dat.data', [235 245], Dat.sampling_frequency, order, 'stop', 'both')';

% Remove <1Hz (present in Patient_2)
% High-pass filter
Dat.data = butterfiltfilt(Dat.data', [0.1 Inf], Dat.sampling_frequency, order)';

end