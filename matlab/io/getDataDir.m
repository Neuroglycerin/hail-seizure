% GETDATADIR Finds the directory of the raw data files on this system

function datdir = getDataDir()

% ** Add the directory of the data on your system to this cell array **
potential_dirs = {...
    '/media/SPARROWHAWK/hail-seizure-data/', ...
    '/media/scott/SPARROWHAWK/hail-seizure-data/', ...
    '/storage/fin/kaggle/hail-seizure-annex', ...
    };

% (If you need to use a directory which exists on other people's systems,
% we can switch on getComputerName instead of looping over the directories)

% Find the folder in the list which exists
for iDir=1:length(potential_dirs)
    if exist(potential_dirs{iDir},'dir')
        datdir = potential_dirs{iDir};
        return;
    end
end

error('No data directory found');

end
