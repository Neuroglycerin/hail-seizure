% Gets the metadata for each file for a given subject and ictal type
% Loads each file and works out which hour of recording it is from
function [fnames, listSegID, listHourID, listSequence] = makeSegMeta(subj, ictyp)

% Get a list of the files for this ictal type
[fnames, mydir, segIDs] = subjtyp2dirs(subj, ictyp);
nFle = numel(fnames);

% Initialise
listSegID    = nan(nFle,1);
listSequence = nan(nFle,1);
listHourID   = nan(nFle,1);

hourID = 1;
lastSegID = 0;
lastSeqnc = 0;
% Loop over all files
for iFle=1:nFle
    % Have to load the entire mat file because it is a structure in a
    % structure
    Dat = loadSegFile(fullfile(mydir,fnames{iFle}));
    % Check the files are incremental!
    if Dat.segID ~= lastSegID+1
        error('Non-incremental segment IDs');
    end
    % Check if we skip from one hour to the next
    if Dat.sequence < lastSeqnc
        hourID = hourID+1;
    end
    % Add to the holding variables
    listSegID(iFle)    = Dat.segID;
    listSequence(iFle) = Dat.sequence;
    listHourID(iFle)   = hourID;
    % Remember for next time
    lastSegID = Dat.segID;
    lastSeqnc = Dat.sequence;
end

end