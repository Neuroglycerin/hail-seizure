% Writes a json file containing a dictionary of metadata for segment files
% The JSON contains a structure fer each segment as follows.
%    Filename: {"subject": , "ictyp": , "segID": , "hourID": , "seqence": , "pseudosolo": }
function writeSegMeta()

metafname = fullfile(getRepoDir(), 'segmentMetadata.json');

subjLst  = subjnames();
ictypLst = {'preictal','interictal','pseudopreictal','pseudointerictal'};

fid = fopen(metafname,'w+');
fprintf(fid,'{\n');

for iSub=1:length(subjLst)
    for iIct=1:length(ictypLst)
        subj = subjLst{iSub};
        ictyp = ictypLst{iIct};
        [fnames, listSegID, listHourID, listSequence, listPseudoBorder] ...
            = makeSegMeta(subj, ictyp);
        for iFle=1:length(fnames)
            fprintf(fid,'\t"%s": {"subject": "%s", "ictyp": "%s", "segID": %.1f, "hourID": %d, "seqence": %.1f, "solopseudo": %d},\n',...
                fnames{iFle}(1:end-4), subj, ictyp, listSegID(iFle), listHourID(iFle), listSequence(iFle), listPseudoBorder(iFle));
        end
    end
end

fprintf(fid,'}');

fclose(fid);

end