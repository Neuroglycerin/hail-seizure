
subjlst = subjnames();

pbgf = {'feat_lmom-3','feat_mvar-GPDC','feat_PSDlogfcorrcoef','pwling1',    'xcorr-ypeak'};
pbgm = {'cln,csp,dwn','cln,ica,dwn',   'cln,ica,dwn',         'cln,ica,dwn','cln,ica,dwn'};

test_set = [120 0.05 0.1 0.25 0.333 0.5];
train_set = [20 0.05 0.1 0.25 0.333 0.5];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parfor iTest = 1:length(test_set)

    outfeatname = ['pbg_test' num2str(test_set(iTest))];
    
    ictypgroupings = {...
        {'preictal','pseudopreictal', 'interictal','pseudointerictal'},...
        {'test'}...
        };
    removebest = true;

    for iSbj=1:numel(subjlst)
        featReduceDimCSP(outfeatname, pbgf, pbgm, subjlst{iSbj}, ictypgroupings, test_set(iTest), removebest);
    end

    for iTrain = 1:length(train_set)
        
        ictypgroupings = {...
            {'preictal','pseudopreictal'},...
            {'interictal','pseudointerictal'}...
            };
        removebest = false;
        
        outfeatname = [outfeatname ',train' num2str(train_set(iTrain))];
        
        for iSbj=1:numel(subjlst)
            featReduceDimCSP(outfeatname, {outfeatname}, {'combo'}, subjlst{iSbj}, ictypgroupings, train_set(iTrain), removebest);
        end
        
    end
end


