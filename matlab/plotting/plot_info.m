function plot_info(featname,modtyp,subjlst,featversion,saveFlag)
    
    if nargin<3 || isempty(subjlst)
        subjlst = subjnames();
    end
    if nargin<4 || isempty(featversion)
        featversion = '';
    end
    if nargin<5 || isempty(saveFlag)
        saveFlag = 1;
    end
    
    linefeatlst = {...
        'feat_psd'
        'feat_psd_logf'
        'feat_coher'
        'feat_coher_logf'
        'feat_pib'
        'feat_FFT'
        };
    
    saveDir = fullfile(getRepoDir(),'plots','featureInfo',[modtyp '_' featname]);
    if saveFlag && ~exist(saveDir,'dir'); mkdir(saveDir); end;
    
    nSub = length(subjlst);
    
    I    = cell(nSub,1); % Information
    Ierr = cell(nSub,1); % Error on information
    Isig = cell(nSub,1); % Relative significance of information
    p    = cell(nSub,1);   % Metadata
    for iSub=1:nSub
        % Load from file
        [I{iSub},Ierr{iSub}] = getInfoFromHDF5(featname, subjlst{iSub}, modtyp, featversion);
        % Merge dims higher than 2 together
        I{iSub} = reshape(I{iSub},size(I{iSub},1),size(I{iSub},2),[]);
        Ierr{iSub} = reshape(Ierr{iSub},size(Ierr{iSub},1),size(Ierr{iSub},2),[]);
        % Find significance of information values
        Isig{iSub} = I{iSub}./Ierr{iSub};
        % Find metadata for this feature
        p{iSub} = getFeatMeta(featname,subjlst{iSub});
    end
    disp('Loaded data');
    
    siz = ndims(I{1});
    s3 = max(cellfun('size',I,3));
    %I2 = cell2matpadNaN(I);
    
    hf = plotHist(I, subjlst, [modtyp '_' featname]);
    if saveFlag;
        saveNme = sprintf('plotHistBasic_%s_%s',modtyp,featname);
        export_fig(hf, fullfile(saveDir,saveNme),'-png');
    end
    
    hf = plotHistInfo(I, Ierr, subjlst, [modtyp '_' featname]);
    if saveFlag;
        saveNme = sprintf('plotHistInfo_%s_%s',modtyp,featname);
        export_fig(hf, fullfile(saveDir,saveNme),'-png');
    end
    
    hf = plotHistSig(Isig, subjlst, [modtyp '_' featname]);
    if saveFlag;
        saveNme = sprintf('plotHistSig_%s_%s',modtyp,featname);
        export_fig(hf, fullfile(saveDir,saveNme),'-png');
    end
    
    if ismember(featname,linefeatlst)
        hf1 = plotHMinfo2(I, subjlst, p, 'Info (bits)', [modtyp '_' featname]);
        hf2 = plotHMinfo2(Isig, subjlst, p, 'InfoSig (sigma)', [modtyp '_' featname]);
        if saveFlag;
            saveNme = sprintf('plotHMinfo_%s_%s',modtyp,featname);
            export_fig(hf1, fullfile(saveDir,saveNme),'-png');
            saveNme = sprintf('plotHMsig_%s_%s',modtyp,featname);
            export_fig(hf2, fullfile(saveDir,saveNme),'-png');
        end
    else
        for iD3=1:s3
            hf1 = 0;
            hf2 = 0;
            if size(I{1},2)>30
                hf1 = plotHMinfoXChn(I, subjlst, p, 'Info (bits)', iD3, [modtyp '_' featname]);
                hf2 = plotHMinfoXChn(Isig, subjlst, p, 'InfoSig (sigma)', iD3, [modtyp '_' featname]);
            end
            if hf1==0
                hf1 = plotHMinfoChn(I, subjlst, 'Info (bits)', iD3, [modtyp '_' featname]);
                hf2 = plotHMinfoChn(Isig, subjlst, 'InfoSig (sigma)', iD3, [modtyp '_' featname]);
            end
            if saveFlag;
                saveNme = sprintf('plotChnInfo_%s_%s_%03d',modtyp,featname,iD3);
                export_fig(hf1, fullfile(saveDir,saveNme),'-png');
                saveNme = sprintf('plotChnSig_%s_%s_%03d',modtyp,featname,iD3);
                export_fig(hf2, fullfile(saveDir,saveNme),'-png');
            end
        end
    end
    
end

function hf = plotHMinfoChn(myI, subjlst, CBLBL, iD3, featname)
    
    nSub = length(myI);
    
    hf = figure('Color',[1 1 1],'Position',[20 20 1200 800]);
    
    for iSub=1:nSub
        ax = subplot(nSub,1,iSub);
        imagesc(squeeze(myI{iSub}(:,:,iD3)));
        colormap(bone(16));
        hcb = colorbar;
        
        if size(myI{iSub},2)>14 && size(myI{iSub},2)<26
            txt = 'Channels';
        elseif size(myI{iSub},2)>90 && size(myI{iSub},2)<300
            txt = 'Channel pairs';
        else
            txt = '';
        end
        caxis([0 mmax(myI{iSub}(:,:,iD3))] + [-1 1]*0.0000001);
        if iSub==nSub;
            xlabel(txt,'Interpreter','none');
        end
        ylabel(subjlst{iSub},'Interpreter','none');
        ylabel(hcb,CBLBL);
        set(ax,'YTick',[]);
        set(ax,'TickDir','out');
        if iSub==1
            title(featname,'Interpreter','none');
        end
    end

end

function hf = plotHMinfoXChn(myI, subjlst, p, CBLBL, iD3, featname)
    
    xlenfig = 1250;
    ylenfig =  800;
    xaxsep  = 0.65;
    yaxsep  = 0.4;
    cblen   = 0.08;
    
    nSub = length(myI);
    if nSub==7;
        nR = 2; nC = 4;
    else
        nC = ceil(sqrt(nSub));
        nR = ceil(nSub/nC);
    end
    
    hf = figure('Color',[1 1 1],'Position',[20 20 xlenfig ylenfig]);
    
    axlen = floor(min(xlenfig/(nC*(1+xaxsep)+1), ylenfig/(nR*(1+yaxsep)+1)));
    
    for iSub=1:nSub
        
        % Reshape so we have a square
        nPair = size(myI{iSub},2);
        
        if ~isfield(p(iSub),'nChn')
            nChn = [];
        else
            nChn = p{iSub}.nChn;
        end
        if isempty(nChn)
            nChn = sqrt(nPair);
            if nChn==floor(nChn)
                % nPair is square, so use square root
            elseif nPair==16*15/2
                nChn = 16;
            elseif nPair==15*14/2
                nChn = 15;
            elseif nPair==24*23/2
                nChn = 24;
            else
                r = roots([1 -1 -2*nPair]);
                nChn = r(r>0);
                if nChn~=floor(nChn)
                    warning('Number of channels is non-integer');
                    hf = 0;
                    return;
                end
            end
        end
        
        myIrshp = zeros(nChn,nChn);
        iChn1 = 1;
        iChn2 = 1;
        for iPair=1:nPair
            if nPair==nChn*nChn
                iChn2 = mod(iPair, nChn)+1;
                iChn1 = ceil(iPair/nPair);
                myIrshp(iChn1,iChn2) = myI{iSub}(1,iPair,iD3);
                myIrshp(iChn2,iChn1) = myIrshp(iChn1,iChn2);
            elseif nPair==nChn*(nChn-1)/2
                iChn2 = iChn2+1;
                if iChn2>nChn
                    iChn1 = iChn1+1;
                    iChn2 = iChn1+1;
                end
                myIrshp(iChn1,iChn2) = myI{iSub}(1,iPair,iD3);
                myIrshp(iChn2,iChn1) = myIrshp(iChn1,iChn2);
            else
                hf = 0;
                return;
            end
        end
        
        % ax = subplot(nSub,1,iSub);
        
        % Determine axes position
        iC = mod((iSub-1),nC)+1;
        iR = ceil(iSub/nC);
        axpos = [...
            axlen*(iC-1)*(1+xaxsep)+axlen/2 ...
            ylenfig-(iR)*axlen*(1+yaxsep)-axlen/2 ...
            axlen ...
            axlen];
        
        ax = axes;
        set(ax,'Units','pixels','Position',axpos);
        
        % Plot
        imagesc(myIrshp);
        colormap(bone(16));
        caxis([0 mmax(myIrshp)] + [-1 1]*0.0000001);
        
        % Bumf
        hcb = colorbar;
        title(subjlst{iSub},'Interpreter','none');
        if iC==1;  ylabel('Channels','Interpreter','none'); end;
        if iR==nR; xlabel('Channels','Interpreter','none'); end;
        ylabel(hcb,CBLBL);
        if iSub==1
            title({featname; subjlst{iSub}},'Interpreter','none');
        end
        
        set(ax,'TickDir','out');
        
        % Position colorbar
        cbpos = axpos;
        cbpos(3) = axlen*cblen;
        cbpos(1) = axpos(1)+axpos(3)+cbpos(3);
        set(hcb,'Units','pixels','Position',cbpos);
        
        % Replace axes in right location
        set(ax,'Units','pixels','Position',axpos);
    end

end

function hf = plotHMinfo2(myI, subjlst, p, CBLBL, featname)
    
    nSub = length(myI);
    
    hf = figure('Color',[1 1 1],'Position',[20 20 1250 1000]);
    
    for iSub=1:nSub
        ax = subplot(nSub,1,iSub);
        
        if strcmp('log',p{iSub}.xscl)
            li = (p{iSub}.x > 0);
            imagesc(log10(p{iSub}.x(li)), 1:size(myI{iSub},2), permute(myI{iSub}(1,:,li),[2 3 1]));
            p{iSub}.xlbl = ['log10 of ' p{iSub}.xlbl];
        else
            imagesc(p{iSub}.x, 1:size(myI{iSub},2), permute(myI{iSub},[2 3 1]));
        end
        
        colormap(bone(16));
        hcb = colorbar;
        
        if size(myI{iSub},2)>14 && size(myI{iSub},2)<26
            txt = 'Channels';
        elseif size(myI{iSub},2)>90 && size(myI{iSub},2)<300
            txt = 'Channel pairs';
        else
            txt = '';
        end
        caxis([0 mmax(myI{iSub})] + [-1 1]*0.0000001);
        if iSub==nSub;
            xlabel(p{iSub}.xlbl,'Interpreter','none');
        end
        ylabel(subjlst{iSub},'Interpreter','none');
        ylabel(hcb,CBLBL);
        if iSub==1
            title(featname,'Interpreter','none');
        end
        set(ax,'TickDir','out');
    end

end

function hf = plotHist(myI, subjlst, featname)
    
    Istp = 0.01; % Information hisogram interval
    
    Imax = max(cellfun(@maxel,myI));
    Imin = min(cellfun(@mmin,myI));
    
    edges = union(0:-Istp:(Imin-Istp*3), 0:Istp:(Imax+Istp*3));

    hf = figure('Color',[1 1 1],'Position',[20 20 500 950]);
    nSub = length(myI);
    for iSub=1:nSub
        ax = subplot(nSub,1,iSub);

        cnt = histc(myI{iSub}(:),edges);
        XLIM = edges([1 end]);
        YLIM = max(cnt) * [-0.1 1.1];
        
        bar(edges,cnt,'histc');
        
        set(ax,'XLim',XLIM,'YLim',YLIM);
        if iSub==1
            title(featname,'Interpreter','none');
        end

        ylabel(subjlst{iSub},'Interpreter','none');
    end

end

function hf = plotHistInfo(myI, Ierr, subjlst, featname)

    Istp = 0.01; % Information hisogram interval
    
    Imax = max(cellfun(@maxel,myI));
    Imin = min(cellfun(@mmin,myI));
    
    %nBin = floor(sqrt(min(cellfun('prodofsize',I)))); % alternative
    
    edges = union(0:-Istp:(Imin-Istp*3), 0:Istp:(Imax+Istp*3));
    ctrs = (edges(1:end-1)+edges(2:end))/2;
    
    hf = figure('Color',[1 1 1],'Position',[20 20 500 950]);
    nSub = length(myI);
    for iSub=1:nSub
        ax = subplot(nSub,1,iSub);
        hold on;

        cnt = histc(myI{iSub}(:),edges);

        XLIM = edges([1 end]);
        YLIM = max(cnt) * [-0.1 1.1];

        plot(XLIM,[0 0],'k');
        plot([0 0],YLIM,'k');

        mdnErr = median(Ierr{iSub}(:));
        
        for iStd=1:3
            plot(mdnErr*iStd*[1 1],YLIM,'r-','Color',[.9 0.3 0.3]);
        end
        for iStd=4:5
            plot(mdnErr*iStd*[1 1],YLIM,'r-','Color',[1 .7 .7]);
        end

        plot(ctrs,cnt(1:end-1),'b','LineWidth',2);

        set(ax,'XLim',XLIM,'YLim',YLIM);
        if iSub==nSub;
            xlabel('Estimated Information (bits)');
        end
        ylabel(subjlst{iSub},'Interpreter','none');
        if iSub==1
            title(featname,'Interpreter','none');
        end
    end
end

function hf = plotHistSig(myI, subjlst, featname)
    
    Imax = max(cellfun(@maxel,myI));
    Imin = min(cellfun(@mmin,myI));
    
    Istp = 1; % Every 1 stdev
    edges = union(0:-Istp:(Imin-Istp*3), 0:Istp:(Imax+Istp*3));
    ctrs = (edges(1:end-1)+edges(2:end))/2;
    
    hf = figure('Color',[1 1 1],'Position',[20 20 500 950]);
    nSub = length(myI);
    for iSub=1:nSub
        ax = subplot(nSub,1,iSub);
        hold on;

        cnt = histc(myI{iSub}(:),edges);

        XLIM = edges([1 end]);
        YLIM = max(cnt) * [-0.1 1.1];

        for iStd=1:3
            plot(iStd*[1 1],YLIM,'r-','Color',[.9 0.3 0.3]);
        end
        for iStd=4:5
            plot(iStd*[1 1],YLIM,'r-','Color',[1 .7 .7]);
        end
        
        plot(XLIM,[0 0],'k');
        plot([0 0],YLIM,'k');

        plot(ctrs,cnt(1:end-1),'b','LineWidth',2);

        set(ax,'XLim',XLIM,'YLim',YLIM);
        if iSub==nSub;
            xlabel('Information Significance (sigmas)');
        end
        ylabel(subjlst{iSub},'Interpreter','none');
        if iSub==1
            title(featname,'Interpreter','none');
        end
    end
end

function p = getFeatMeta(featname,subj)

    % Load and process data for one file
    [fnames, mydir] = subjtyp2dirs(subj, 'pre', 'raw');
    if isempty(fnames);
        p = struct([]);
        return;
    end;
    Dat = loadSegFile(fullfile(mydir,fnames{1}));
    p.nChn = size(Dat.data,1);
    fnc = str2func(featname);
    [~,outparams] = fnc(Dat);
    switch featname
        case {'feat_FFT'}
            p.x = outparams.f;
            p.xscl = 'linear';
            p.xlbl = 'Freq (Hz)';
            p.CBLBLl = '';
        case {'feat_psd','feat_coher'}
            p.x = outparams.f;
            p.xscl = 'linear';
            p.xlbl = 'Freq (Hz)';
            p.CBLBLl = 'Power (dB)';
        case {'feat_psd_logf','feat_coher_logf'}
            p.x = mean(outparams.bandEdges,1);
            p.xscl = 'log';
            p.xlbl = 'Freq (Hz)';
            p.CBLBLl = 'Power (dB)';
        case {'feat_pib'}
            bndnms = fieldnames(outparams.bands);
            p.x = nan(1,length(bndnms));
            for i=1:length(bndnms)
                p.x(i) = mean(outparams.bands.(bndnms{i}));
            end
            p.xscl = 'log';
            p.xlbl = 'Freq (Hz)';
            p.CBLBLl = 'Power (raw)';
        otherwise
            p.x = [];
            p.xscl = 'linear';
            p.xlbl = '';
            p.CBLBLl = '';
    end
    
end