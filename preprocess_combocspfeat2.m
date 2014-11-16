addpath(genpath('matlab'))

subjlst = subjnames();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

outfeatname = 'allfeat_test0.1';
separate_from_test = true;
ncomponents = 0.1;

featnames = {
                'feat_act',
                'feat_ampcorrcoef-alpha',
                'feat_ampcorrcoef-alpha-eig',
                'feat_ampcorrcoef-beta',
                'feat_ampcorrcoef-beta-eig',
                'feat_ampcorrcoef-delta',
                'feat_ampcorrcoef-delta-eig',
                'feat_ampcorrcoef-high_gamma',
                'feat_ampcorrcoef-high_gamma-eig',
                'feat_ampcorrcoef-low_gamma',
                'feat_ampcorrcoef-low_gamma-eig',
                'feat_ampcorrcoef-theta',
                'feat_ampcorrcoef-theta-eig',
                'feat_coher_logf',
                'feat_corrcoef',
                'feat_corrcoefeig',
                'feat_cov',
                'feat_emvar-ARF',
                'feat_emvar-eCOHphs',
                'feat_emvar-eGphs',
                'feat_emvar-eSphs',
                'feat_emvar-PDCphs',
                'feat_emvar-COHphs',
                'feat_emvar-eCOH',
                'feat_emvar-eG',
                'feat_emvar-eS',
                'feat_emvar-PDC',
                'feat_emvar-COH',
                'feat_emvar-eDCphs',
                'feat_emvar-ePCOHphs',
                'feat_emvar-GPDCphs',
                'feat_emvar-Pphs',
                'feat_emvar-DCphs',
                'feat_emvar-eDC',
                'feat_emvar-ePCOH',
                'feat_emvar-GPDC',
                'feat_emvar-P',
                'feat_emvar-DC',
                'feat_emvar-eDDCphs',
                'feat_emvar-ePDCphs',
                'feat_emvar-Hphs',
                'feat_emvar-Sphs',
                'feat_emvar-DTFphs',
                'feat_emvar-eDDC',
                'feat_emvar-ePDC',
                'feat_emvar-H',
                'feat_emvar-S',
                'feat_emvar-DTF',
                'feat_emvar-eDPDCphs',
                'feat_emvar-ePphs',
                'feat_emvar-PCOHphs',
                'feat_emvar-eARF',
                'feat_emvar-eDPDC',
                'feat_emvar-eP',
                'feat_emvar-PCOH',
                'feat_FFT',
                'feat_FFTcorrcoef',
                'feat_FFTcorrcoefeig',
                'feat_ilingam-causalindex',
                'feat_ilingam-connweights',
                'feat_lmom-1',
                'feat_lmom-2',
                'feat_lmom-3',
                'feat_lmom-4',
                'feat_lmom-5',
                'feat_lmom-6',
                'feat_phase-alpha-dif',
                'feat_phase-alpha-sync',
                'feat_phase-beta-dif',
                'feat_phase-beta-sync',
                'feat_phase-delta-dif',
                'feat_phase-delta-sync',
                'feat_phase-high_gamma-dif',
                'feat_phase-high_gamma-sync',
                'feat_phase-low_gamma-dif',
                'feat_phase-low_gamma-sync',
                'feat_phase-theta-dif',
                'feat_phase-theta-sync',
                'feat_pib',
                'feat_pib_ratio',
                'feat_pib_ratioBB',
                'feat_PSDcorrcoef',
                'feat_PSDcorrcoefeig',
                'feat_psd_logf',
                'feat_psd_logfBB',
                'feat_PSDlogfcorrcoef',
                'feat_PSDlogfcorrcoefeig',
                'feat_pwling1',
                'feat_pwling2',
                'feat_pwling4',
                'feat_pwling5',
                'feat_spearman',
                'feat_var',
                'feat_xcorr-tpeak',
                'feat_xcorr-twidth',
                'feat_xcorr-ypeak'
                };
            
modtyps = {};
for i=1:length(featnames)
    modtyps{i} = 'cln,ica,dwn';
end

if ~separate_from_test
    ictypgroupings{1} = {'preictal','pseudopreictal'};
    ictypgroupings{2} = {'interictal','pseudointerictal'};
    removebest = false;
else
    ictypgroupings{1} = {'preictal','pseudopreictal', 'interictal','pseudointerictal'};
    ictypgroupings{2} = {'test'};
    removebest = true;
end

for iSbj=1:numel(subjlst)
    featReduceDimCSP(outfeatname, featnames, modtyps, subjlst{iSbj}, ictypgroupings, ncomponents, removebest);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

outfeatname2 = [outfeatname ',train0.1'];
separate_from_test = false;
ncomponents = 0.1;

featnames = {'feat_lmom-3'};
modtyps = {'cln,csp,dwn'};


if ~separate_from_test
    ictypgroupings{1} = {'preictal','pseudopreictal'};
    ictypgroupings{2} = {'interictal','pseudointerictal'};
    removebest = false;
else
    ictypgroupings{1} = {'preictal','pseudopreictal', 'interictal','pseudointerictal'};
    ictypgroupings{2} = {'test'};
    removebest = true;
end

for iSbj=1:numel(subjlst)
    featReduceDimCSP(outfeatname2, {outfeatname}, {'combo'}, subjlst{iSbj}, ictypgroupings, ncomponents, removebest);
end
