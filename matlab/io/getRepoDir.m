% If it is on the path, finds the SETTINGS.json file in the root folder of
% the repository

function repodir = getRepoDir()

settingsfname = 'SETTINGS.json';
repodir = fileparts(which(settingsfname));

end