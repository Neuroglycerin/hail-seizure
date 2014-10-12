function [I,Ierr] = getInfoFromHDF5(featname, subj, modtyp, featversion)

% Default inputs
if nargin<5; featversion = ''; end;

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Main --------------------------------------------------------------------
% Use current version by default
if isempty(featversion)
    settings = json.read(settingsfname);
    featversion = settings.VERSION;
end
% Work out h5 filename
h5fnme = getFeatH5fname(featname, modtyp, featversion);

if ~exist(h5fnme,'file');
    error('HDF5 file does not exist');
end

I = h5read(h5fnme, ['/' subj '/', 'MI']);
Ierr = h5read(h5fnme, ['/' subj '/', 'MIerr']);

end