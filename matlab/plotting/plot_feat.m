% plot_feat
% Plot summary of a feature
% saveFlag=0: Show but don't save
% saveFlag=1: Show and save
% saveFlag=-1: Don't show but do save

function plot_feat(featname, subj, modtyp,saveFlag)
    
    if nargin<4
        saveFlag = -1;
    end

    % Load data
    feat0 = getFeatFromHDF5(featname, subj, 'interictal', modtyp);
    feat1 = getFeatFromHDF5(featname, subj, 'preictal', modtyp);

    % if length(siz)>3; error('too many dims'); end;
    
    saveDir = fullfile(getRepoDir(),'plots','featureDistribution');
    if ~exist(saveDir,'dir'); mkdir(saveDir); end;
    
    plotLine(feat0,feat1,saveFlag,saveDir,featname,subj,modtyp);
    plotDistr(feat0,feat1,saveFlag,saveDir,featname,subj,modtyp);
    
end

% Data distribution
function plotDistr(feat0,feat1,saveFlag,saveDir,featname,subj,modtyp)
    
    if saveFlag;
        saveDir = fullfile(saveDir,[modtyp '_' featname],'distribution',subj);
        if ~exist(saveDir,'dir'); mkdir(saveDir); end;
    end
    
    % Reshape so we lose any dimensions higher than 3
    feat0 = reshape(feat0,size(feat0,1),size(feat0,2),[]);
    feat1 = reshape(feat1,size(feat1,1),size(feat1,2),[]);

    % Check sizing
    siz0 = size(feat0);
    siz1 = size(feat1);
    if ~isequal(siz0(2:end),siz1(2:end)); error('unbalanced sizes'); end;
    sizr = siz0;
    sizr(1) = max(siz0(1),siz1(1));
    if length(sizr)<3; sizr(3) = 1; end;
    
    for iD3=1:sizr(3)
        hf = figure('Position',[30 30 950 950],'Color',[1 1 1]);
        if saveFlag<0; set(hf,'Visible','Off'); end;
        % Merge pre and inter together
        ff = nan([sizr(1) 3*sizr(2)]);
        ff(1:siz0(1), 1:3:end) = feat0(:,:,iD3);
        ff(1:siz1(1), 2:3:end) = feat1(:,:,iD3);
        boxplot(ff,'plotstyle','compact','symbol','.','colors','brw');
        if sizr(2)>14 && sizr(2)<25
                xlabel('Channels');
        elseif sizr(2)>90 && sizr(2)<300
            xlabel('Channel pair combinations');
        end
        % Save
        if saveFlag;
            saveNme = sprintf('plotdstrbn_%s_%s_%s_%03d',modtyp,featname,subj,iD3);
            export_fig(hf, fullfile(saveDir,saveNme),'-png');
            close(hf);
        end
    end
end

function plotLine(feat0,feat1,saveFlag,saveDir,featname,subj,modtyp)
    
    saveDir = fullfile(saveDir,[modtyp '_' featname],'line');
    
    linefeatlst = {...
        'feat_psd'
        'feat_psd_logf'
        'feat_coher'
        'feat_coher_logf'
        'feat_pib'
        'feat_FFT'
        };
    
    if ~ismember(featname, linefeatlst)
        return;
    end
    if ndims(feat0)<3
        return;
    end
    
    if saveFlag && ~exist(saveDir,'dir');
        mkdir(saveDir);
    end;
    
    [x,nChn,xscl,xlbl,ylbl] = getFeatMeta(featname,subj);
    
    % Reshape so we lose any dimensions higher than 4
    feat0 = reshape(feat0,size(feat0,1),size(feat0,2),size(feat0,3),[]);
    feat1 = reshape(feat1,size(feat1,1),size(feat1,2),size(feat1,3),[]);
    
    % Take mean, median and standard deviation only
    mu0 = mean(feat0,1);
    se0 = std(feat0,1)/sqrt(size(feat0,1));
    mu1 = mean(feat1,1);
    se1 = std(feat1,1)/sqrt(size(feat1,1));
    
    md0 = median(feat0,1);
    uq0 = prctile(feat0,75,1);
    lq0 = prctile(feat0,25,1);
    md1 = median(feat1,1);
    uq1 = prctile(feat1,75,1);
    lq1 = prctile(feat1,25,1);
    
    % Check sizing
    siz0 = size(feat0);
    siz1 = size(feat1);
    if ~isequal(siz0(2:end),siz1(2:end)); error('unbalanced sizes'); end;
    sizr = siz0;
    sizr(1) = max(siz0(1),siz1(1));
    if length(sizr)<4; sizr(4) = 1; end;
    
    for iStt = [1 2]
        if iStt==1;
            y0 = mu0;
            b0 = 3*se0;
            y1 = mu1;
            b1 = 3*se1;
        else
            y0 = md0;
            b0 = cat(5, (uq0-md0), (md0-lq0));
            y1 = md1;
            b1 = cat(5, (uq1-md1), (md1-lq1));
        end
        for iD4=1:sizr(4)
            hf = figure('Position',[30 30 950 950],'Color',[1 1 1]);
            if saveFlag<0; set(hf,'Visible','Off'); end;
            iChn1 = 1;
            iChn2 = 1;
%             
%             if iStt==1;
%                 YLIM = [...
%                     mmin(...
%                         [y0(1,:,:,iD4)-b0(1,:,:,iD4); ...
%                          y1(1,:,:,iD4)-b1(1,:,:,iD4)]), ...
%                     mmax(...
%                         [y0(1,:,:,iD4)+b0(1,:,:,iD4); ...
%                          y1(1,:,:,iD4)+b1(1,:,:,iD4)]) ...
%                     ];
%             else
%                 YLIM = [...
%                     mmin([lq0(1,:,:,iD4); lq1(1,:,:,iD4)]), ...
%                     mmax([uq0(1,:,:,iD4); uq1(1,:,:,iD4)])  ...
%                     ];
%             end
            
            for iD2=1:sizr(2)
                
            if iStt==1;
                YLIM = [...
                    mmin(...
                        [y0(1,iD2,:,iD4)-b0(1,iD2,:,iD4); ...
                         y1(1,iD2,:,iD4)-b1(1,iD2,:,iD4)]), ...
                    mmax(...
                        [y0(1,iD2,:,iD4)+b0(1,iD2,:,iD4); ...
                         y1(1,iD2,:,iD4)+b1(1,iD2,:,iD4)]) ...
                    ];
            else
                YLIM = [...
                    mmin([lq0(1,iD2,:,iD4); lq1(1,iD2,:,iD4)]), ...
                    mmax([uq0(1,iD2,:,iD4); uq1(1,iD2,:,iD4)])  ...
                    ];
            end
                
                if true
                    nC = ceil(sqrt(sizr(2)));
                    nR = ceil(sizr(2)/nC);
                    subplot(nR,nC,iD2);
                    lftedg = (mod(iD2,nC)==1);
                    btmedg = (iD2>sizr(2)-nC);
                    
                elseif sizr(2)==nChn
                    subplot(nChn,1,iD2);
                    
                elseif sizr(2)==nChn*nChn
                    iChn2 = mod(iD2,nChn)+1;
                    iChn1 = ceil(iD2/sizr(2));
                    subplot(nChn,nChn,(iChn1-1)*nChn+iChn2);
                    
                elseif sizr(2)==nChn*(nChn-1)/2
                    iChn2 = iChn2+1;
                    if iChn2>nChn
                        iChn1 = iChn1+1;
                        iChn2 = iChn1+1;
                    end
                    subplot(nChn,nChn,(iChn1-1)*nChn+iChn2);
                    
                else
                    subplot(1,sizr(2),iD2);
                    
                end
                
                hold on;
                
                if sizr(2)<50
                    [hl0, hp0] = boundedline(x, squeeze(y0(1,iD2,:,iD4)), squeeze(b0(1,iD2,:,iD4,:)), 'b');
                    outlinebounds(hl0,hp0);

                    [hl1, hp1] = boundedline(x, squeeze(y1(1,iD2,:,iD4)), squeeze(b1(1,iD2,:,iD4,:)), 'r');
                    outlinebounds(hl1,hp1);
                    
                    uistack(hp1,'bottom');
                    uistack(hp0,'bottom');
                else
                    plot(x, squeeze(y0(1,iD2,:,iD4)), 'b-');
                    for iD5=1:size(b0,5)
                        plot(x, squeeze(b0(1,iD2,:,iD4,iD5)), 'b-');
                    end
                    
                    plot(x, squeeze(y1(1,iD2,:,iD4)), 'r-');
                    for iD5=1:size(b1,5)
                        plot(x, squeeze(b1(1,iD2,:,iD4,iD5)), 'r-');
                    end
                end
                
                
                set(gca,'XScale',xscl);
                if btmedg;
                    xlabel(xlbl);
                else
                    set(gca,'XTick',[]);
                    set(gca,'XColor',[1 1 1]);
                end;
                if lftedg;
                    ylabel(ylbl);
                else
                    set(gca,'YTick',[]);
                    set(gca,'YColor',[1 1 1]);
                end;
                xlim(x([1 end]));
                ylim(YLIM);
                
            end
            
            % Save
            if saveFlag;
                if iStt==1
                    saveNme = sprintf('plotlineMean_%s_%s_%s_%03d',modtyp,featname,subj,iD4);
                else
                    saveNme = sprintf('plotlineMedian_%s_%s_%s_%03d',modtyp,featname,subj,iD4);
                end
                export_fig(hf, fullfile(saveDir,saveNme),'-png');
                close(hf);
            end
        end
    end
end

function [x,nChn,xscl,xlbl,ylbl] = getFeatMeta(featname,subj)

    % Load and process data for one file
    [fnames, mydir] = subjtyp2dirs(subj, 'pre', 'raw');
    if isempty(fnames); return; end;
    Dat = loadSegFile(fullfile(mydir,fnames{1}));
    nChn = size(Dat.data,1);
    fnc = str2func(featname);
    [~,outparams] = fnc(Dat);
    switch featname
        case {'feat_FFT'}
            x = outparams.f;
            xscl = 'linear';
            xlbl = 'Freq (Hz)';
            ylbl = '';
        case {'feat_psd','feat_coher'}
            x = outparams.f;
            xscl = 'linear';
            xlbl = 'Freq (Hz)';
            ylbl = 'Power (dB)';
        case {'feat_psd_logf','feat_coher_logf'}
            x = mean(outparams.bandEdges,1);
            xscl = 'log';
            xlbl = 'Freq (Hz)';
            ylbl = 'Power (dB)';
        case {'feat_pib'}
            bndnms = fieldnames(outparams.bands);
            x = nan(1,length(bndnms));
            for i=1:length(bndnms)
                x(i) = mean(outparams.bands.(bndnms{i}));
            end
            xscl = 'log';
            xlbl = 'Freq (Hz)';
            ylbl = 'Power (raw)';
        otherwise
            error('Line feature name mismatch');
    end
    
end