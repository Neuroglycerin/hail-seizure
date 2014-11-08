% Gets the metadata for each file for a given subject and ictal type
% Loads each file and works out which hour of recording it is from
function [fnames, listSegID, listHourID, listSequence, listPseudoBorder] ...
    = makeSegMeta(subj, ictyp)

% Check if we are generating pseudo training data
if strcmp('pseudopreictal',ictyp)
    ictyp = 'preictal';
    ispseudo = true;
elseif strcmp('pseudointerictal',ictyp)
    ictyp = 'interictal';
    ispseudo = true;
else
    ispseudo = false;
end
% Get a list of the files for this ictal type
[fnames, mydir, segIDs] = subjtyp2dirs(subj, ictyp);
nFle = numel(fnames);

% Initialise
listSegID    = nan(nFle,1);
listSequence = nan(nFle,1);
listHourID   = nan(nFle,1);
listPseudoBorder = nan(nFle,1);

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
    listPseudoBorder(iFle) = 0;
    % PseudoData
    if ispseudo
        fnames{iFle} = strrep(fnames{iFle},ictyp,['pseudo' ictyp]);
        listSegID(iFle) = listSegID(iFle) + 0.5;
        listSequence(iFle) = listSequence(iFle) + 0.5;
    end
    % Check the pseudodata can merge
    if Dat.sequence~=lastSeqnc+1 && ispseudo && iFle>1
        listPseudoBorder(iFle-1) = 1;
    end
    % Remember for next time
    lastSegID = Dat.segID;
    lastSeqnc = Dat.sequence;
end
% Final pseudo is on sequence border
if ispseudo
    listPseudoBorder(nFle) = 1;
end

end