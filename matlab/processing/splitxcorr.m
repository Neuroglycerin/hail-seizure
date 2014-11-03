function splitxcorr(nPrt)
    
    feat_str = 'feat_xcorr';
    if nPrt>1
        feat_str = [num2str(nPrt) feat_str];
    end
    featversion = '_v2';
    
    featnames = cell(1,1,3);
    featnames{1} = 'ypeak';
    featnames{2} = 'tpeak';
    featnames{3} = 'twidth';
    
    
    subjlst = subjnames();
    modtyplst = {'raw','ica'};
    ictyplst = {'preictal'; 'interictal'; 'test'; 'pseudopreictal'; 'pseudointerictal'};
    
    
    for iMod = 1:numel(modtyplst)
        for iSub=1:numel(subjlst)
            for iIct=1:numel(ictyplst)
                fprintf('%s %s %s\n',subjlst{iSub}, ictyplst{iIct}, modtyplst{iMod});
                splitSubFeatAddToHDF5(feat_str, subjlst{iSub}, ictyplst{iIct},...
                    modtyplst{iMod}, featversion, featnames);
            end
        end
    end
end
 
function splitSubFeatAddToHDF5(feat_str, subj, ictyp, modtyp, featversion, featnames)

    featM = getFeatFromHDF5(feat_str, subj, ictyp, modtyp, featversion, true);
    
    % Find which dimension is across multiple subfeatures
    siz = size(featnames);
    dim = find(siz~=1);
    
    % Make a cell to take everything from the array
    allcell = cell(size(siz));
    for iDim=1:length(siz)
        allcell{iDim} = ':';
    end
    % Separate out the subfeatures and save in separate files
    for iCut=1:length(featnames);
        pp = allcell;
        pp{dim} = iCut;
        subfeatM = cell(size(featM));
        subfeatM(:,1) = featM(:,1);
        for iFle=1:size(featM,1)
            subfeatM{iFle,2} = featM{iFle,2}(pp{:});
        end
        addToHDF5(subfeatM, subj, ictyp, [feat_str '-' featnames{iCut}], modtyp);
    end
    
end