
subjlst = subjnames();
modtyps = {'raw','ica'};
modtyps = {'raw'};

for iSub = 4:numel(subjlst)
    for iMod = 1:numel(modtyps)
        fprintf('%s: %s %s %d\n', datestr(now,30), subjlst{iSub}, modtyps{iMod});
        tic;
        avgorder = saveMVARorder(subjlst{iSub}, modtyps{iMod});
        fprintf('%s: (%.0f)sec %s %s %d\n', datestr(now,30), toc, subjlst{iSub}, modtyps{iMod}, avgorder);
    end
end