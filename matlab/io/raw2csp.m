function Dat = raw2csp(Dat, subj, fullmodlst)

% Parameters --------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Main --------------------------------------------------------------------
% Load the settings file
settings = json.read(settingsfname);

% Versions 3 and higher load precleaned files from disk
if ~strcmp(settings.VERSION,'_v1') && ~strcmp(settings.VERSION,'_v2')
    fullmodlst = strrep(fullmodlst,'cln','precln');
end

% Look up where the settings file is: that is the root directory and
% paths inside it are relative to its locations
distrodir = fileparts(which(settingsfname));

% This is the path we expect the weights to be saved at
mydir = fullfile(distrodir,settings.MODEL_PATH);

k = strfind(fullmodlst,'csp');

if isempty(k)
    error('Full mod list does not contain ''csp''');
end

if k==1
    % Use the regular weights
    Wfname = ['csp_weights_' subj '.mat'];
elseif ~isempty(strfind(fullmodlst(1:k-1),'cln')) &&  ~isempty(strfind(fullmodlst(k:end),'dwn'))
    % Special case: use the cleaned and downsampled weights if we are now
    % cleaning before ICA and downsampling afterward. We do this because
    % the downsampling step involves further cleaning of high frequency
    % artefacts
    Wfname = ['csp_weights_' subj '_' fullmodlst(1:k-2) ',dwn.mat'];
else
    % Use the weights where ICA is done after these preprocessing steps
    Wfname = ['csp_weights_' subj '_' fullmodlst(1:k-2) '.mat'];
end

Wfnamefull = fullfile(mydir,Wfname);

% Check the file exists
if ~exist(Wfnamefull,'file');
    error('CSP demixing weights not present');
end

% Load the Weight matrix from the file
Tmp = load(Wfnamefull);
W = Tmp.W;

% Use the weights to separate the components
Dat.data = W * Dat.data;

end
