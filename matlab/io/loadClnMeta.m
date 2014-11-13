function ClnMeta = loadClnMeta(subj)

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% -------------------------------------------------------------------------
% Load the settings file
settings = json.read(settingsfname);

% Work out the file name
mydir = fullfile(getRepoDir(), settings.MODEL_PATH);
Wfname = ['cleaningmetadata_' subj];

Wfnamefull = fullfile(mydir,Wfname);

% Load the metadata
ClnMeta = load(Wfnamefull);

end