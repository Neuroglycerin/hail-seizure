function Dat = raw2ica(Dat, subj, fullmodlst)

% Parameters --------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Main --------------------------------------------------------------------
% Load the settings file
settings = json.read(settingsfname);

% Look up where the settings file is: that is the root directory and
% paths inside it are relative to its locations
distrodir = fileparts(which(settingsfname));

% This is the path we expect the weights to be saved at
mydir = fullfile(distrodir,settings.MODEL_PATH);


if ~isempty(strfind(fullmodlst,'cln')) && ~isempty(strfind(fullmodlst,'dwn'))
    % If it is cleaned, use the weights from the cleaned & downsampled version
    Wfname = ['ica_weights_' subj '_cln,dwn.mat'];
elseif ~isempty(strfind(fullmodlst,'cln'))
    % If it is cleaned, use the weights from the cleaned version
    Wfname = ['ica_weights_' subj '_cln.mat'];
else
    % Use the regular weights
    Wfname = ['ica_weights_' subj '.mat'];
end

Wfnamefull = fullfile(mydir,Wfname);

% Check the file exists
if ~exist(Wfnamefull,'file');
    error('ICA demixing weights not present');
end

% Load the Weight matrix from the file
Tmp = load(Wfnamefull);
W = Tmp.W;

% Use the weights to separate the components
Dat.data = W * Dat.data;

end