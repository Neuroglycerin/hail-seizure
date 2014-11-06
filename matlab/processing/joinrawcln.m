function joinrawcln(feat_str, modtyp, featversion)
    
    settings = json.read('SETTINGS.json');
    
    if nargin<3
        featversion = settings.VERSION;
    end
    
    if ~strcmp(featversion,settings.VERSION)
        error('Inconsistent versions');
    end
    
    subjlst = {'Dog_1','Dog_2','Dog_3','Dog_4','Dog_5'};
    
    ictyplst = {'preictal'; 'interictal'; 'test'; 'pseudopreictal'; 'pseudointerictal'};
    
    h5fnme_raw = getFeatH5fname(feat_str, modtyp, settings.VERSION);
    if ~exist(h5fnme_raw,'file');
        error('joinrawcln:NoRawFile','Raw feature file does not exist');
    end
    h5fnme_cln = getFeatH5fname(feat_str, ['cln,' modtyp ',dwn'], settings.VERSION);
    if ~exist(h5fnme_cln,'file');
        error('joinrawcln:NoClnFile','Clean feature file does not exist');
    end
    
    for iSub=1:numel(subjlst)
        if strcmp(subjlst{iSub},'Patient_1') || strcmp(subjlst{iSub},'Patient_2');
            fprintf('Should not override Patients 1 and 2\n');
            continue;
        end
        for iIct=1:numel(ictyplst)
            try
                % Need to get a list of all the datasets
                scrapeH5datasets(h5info(h5fnme_cln), subjlst{iSub}, ictyplst{iIct});
                fprintf('\tFeature already present in cln for %s %s\n',subjlst{iSub},ictyplst{iIct});
                fprintf('Here 1');
                continue;
            catch ME
                if ~strcmp(ME.identifier,'getFeatFromHDF5:missingSubject') ...
                        && ~strcmp(ME.identifier,'getFeatFromHDF5:missingIctal')
                    rethrow(ME);
                end
            end
            featM = getFeatFromHDF5(feat_str, subjlst{iSub}, ictyplst{iIct}, modtyp, featversion, true);
            fprintf('\tLoaded %s %s %s,...', feat_str, subjlst{iSub}, ictyplst{iIct});
            addToHDF5(featM, subjlst{iSub}, ictyplst{iIct}, feat_str, ['cln,' modtyp ',dwn']);
            fprintf(' written %s %s %s\n', feat_str, subjlst{iSub}, ictyplst{iIct});
        end
    end
    
end
