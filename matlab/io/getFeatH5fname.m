function h5fnme = getFeatH5fname(featname, modtyp, featversion)

% Declarations ------------------------------------------------------------
settingsfname = 'SETTINGS.json';

% Main --------------------------------------------------------------------
% Work out h5 filename
settings = json.read(settingsfname);
h5fnme = [modtyp '_' featname '_' featversion '.h5'];
h5fnme = fullfile(getRepoDir(), settings.TRAIN_DATA_PATH, h5fnme);

end