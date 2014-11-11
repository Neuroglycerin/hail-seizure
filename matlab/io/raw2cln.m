function Dat = raw2cln(Dat)

% Parameters --------------------------------------------------------------
settingsfname = 'SETTINGS.json';
order = 2;

% Main --------------------------------------------------------------------
% Load the settings file
settings = json.read(settingsfname);

% =========================================================================
% Versions 3 and higher load precleaned files from disk
if ~strcmp(settings.VERSION,'_v1') && ~strcmp(settings.VERSION,'_v2')
    ClnMeta = loadClnMeta(subj);
    if ClnMeta.needcln && ~isfield(Dat,'iscln')
        error('You are in version %s. You should load the pre-cleaned file',settings.VERSION);
    end
    if ~ClnMeta.needcln && isfield(Dat,'iscln') && Dat.iscln
        error('You loaded pre-cleaned data when it did not need cleaning');
    end
    % If it was pre-cleaned and supposed to be, we don't need to do anything
    % If it wasn't pre-cleaned and isn't supposed to be, we don't need to do anything
    return;
end
% =========================================================================

% =========================================================================
% Versions 2 or lower run this code

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