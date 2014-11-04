function Dat = raw2dwn(Dat)

% Downsample so we have a common sampling frequency for all sessions

% Parameters
order = 2;
target_fs = 399.609756097560989474;
% target_datalength = 239766;

if Dat.fs<1.5*target_fs
    return;
end

fsratio = Dat.fs/target_fs;
fsratio = round(fsratio*2)/2; % Round to nearest half

if mod(fsratio,1)~=0
    % Upsample by a factor of 2 (if necessary)
    Dat.data = upsample(Dat.data',2)';
    % Double the ratio so we downsample correctly
    fsratio = fsratio*2;
    Dat.fs = Dat.fs*2;
end

% Smooth to target Nyquist frequency with low-pass filter
Dat.data = butterfiltfilt(Dat.data', [0 Dat.fs/fsratio/2], Dat.fs, order)';

% Downsample as appropriate (by a factor of 25)
Dat.data = downsample(Dat.data',fsratio)';
Dat.fs = Dat.fs/fsratio;

end