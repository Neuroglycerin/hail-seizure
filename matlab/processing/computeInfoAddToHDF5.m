function computeInfoAddToHDF5(featname, subj, modtyp, featversion)

% Input handling ----------------------------------------------------------
if nargin<4; featversion=''; end;

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Use current version by default
if isempty(featversion)
    settings = json.read(settingsfname);
    featversion = settings.VERSION;
end

% Load up the data and compute the information content of each feature
% vector element
[I,Ierr] = getFeatComputeInfo(featname, subj, modtyp, featversion);

% Write to the HDF5 file
h5fnme = getFeatH5fname(featname, modtyp, featversion);
h5writePlus(h5fnme, strcat('/', subj, '/', 'MI')   , I);
h5writePlus(h5fnme, strcat('/', subj, '/', 'MIerr'), Ierr);

end