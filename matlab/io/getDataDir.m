% GETDATADIR Finds the directory of the raw data files on this system

function datdir = getDataDir()

settings = json.read('SETTINGS.json');

potential_dirs = settings.RAW_DATA_DIRS;


% could just use a 'data' folder within the repo for everything and have it
% just contain symlinks to wherever the data is actually stored.

% Find the folder in the list which exists
for iDir=1:length(potential_dirs)
    if exist(potential_dirs{iDir},'dir')
        datdir = potential_dirs{iDir};
        return;
    end
end

error('No data directory found');

end
