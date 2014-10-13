
featversion = '';

featname = 'feat_psd_logf';
modtyp = 'raw';

%%
subjlst = subjnames;
nSub = length(subjlst);

I    = cell(nSub,1);
Ierr = cell(nSub,1);
for iSub=1:nSub
    [I{iSub},Ierr{iSub}] = getInfoFromHDF5(featname, subjlst{iSub}, modtyp, featversion);
end

s3 = max(cellfun('size',I,3));
Imax = max(cellfun(@maxel,I));
Imin = min(cellfun(@mmin,I));

I2 = cell2matpadNaN(I);

%%
hf = figure('Position',[20 20 500 950]);
for iSub=1:nSub
    subplot(nSub,1,iSub);
    
    %hist(I{iSub}(:),30);
    
    siz = size(I{iSub});
    myI = nan(siz(1), siz(2), s3);
    myI(:,:,1:siz(3)) = I{iSub};
    
    imagesc(permute(myI,[2 3 1]));
    colormap(bone);
    hcb = colorbar;
    yltxt = subjlst{iSub};
    if size(myI,2)>14 && size(myI,2)<25
        yltxt = {yltxt; 'Channels'};
    elseif size(myI,2)>90 && size(myI,2)<300
        yltxt = {yltxt; 'Channel pair combinations'};
    end
    ylabel(yltxt,'Interpreter','none');
    ylabel(hcb,'Info (bits)');
    
    caxis([0 max(myI(:))]);
    % caxis([0 Imax]);
    
end

%%
hf = figure('Position',[20 20 500 950]);
for iSub=1:nSub
    subplot(nSub,1,iSub);
    
    %hist(I{iSub}(:),30);
    
    siz = size(I{iSub});
    myI = nan(siz(1), siz(2), s3);
    myI(:,:,1:siz(3)) = I{iSub}./Ierr{iSub};
    
    imagesc(permute(myI,[2 3 1]));
    colormap(bone);
    hcb = colorbar;
    if size(myI,2)>14 && size(myI,2)<25
        ylabel('Channels');
    elseif size(myI,2)>90 && size(myI,2)<300
        ylabel('Channel pair combinations');
    end
    ylabel(hcb,'Info sig (sigma)');
    
    caxis([0 max(myI(:))]);
    % caxis([0 Imax]);
    
end

%%
%nBin = floor(sqrt(min(cellfun('prodofsize',I))));
Istp = 0.01;
edges = union(0:-Istp:(Imin-Istp), 0:Istp:(Imax+Istp));

hf = figure('Position',[20 20 500 950]);
for iSub=1:nSub
    subplot(nSub,1,iSub);
    
    cnt = histc(I{iSub}(:),edges);
    bar(edges,cnt,'histc');
    
end

%%
%nBin = floor(sqrt(min(cellfun('prodofsize',I))));
Istp = 0.01;
edges = union(0:-Istp:(Imin-Istp), 0:Istp:(Imax+Istp));
ctrs = (edges(1:end-1)+edges(2:end))/2;
hf = figure('Position',[20 20 500 950]);
for iSub=1:nSub
    ax = subplot(nSub,1,iSub);
    hold on;
    
    cnt = histc(I{iSub}(:),edges);
    
    XLIM = edges([1 end]);
    YLIM = max(cnt) * [-0.1 1.1];
    
    plot(XLIM,[0 0],'k');
    plot([0 0],YLIM,'k');
    
    mdnErr = median(Ierr{iSub}(:));
    for iStd=1:3
        plot(mdnErr*iStd*[1 1],YLIM,'r');
    end
    
    plot(ctrs,cnt(1:end-1),'b');
    
    set(ax,'XLim',XLIM,'YLim',YLIM);
    if iSub==nSub;
        xlabel('Estimated Information (bits)');
    end
end

%%

Istp = .5;
edges = union(0:-Istp:-3, 0:Istp:30);
ctrs = (edges(1:end-1)+edges(2:end))/2;
hf = figure('Position',[20 20 500 950]);
for iSub=1:nSub
    ax = subplot(nSub,1,iSub);
    hold on;
    
    cnt = histc(I{iSub}(:)./Ierr{iSub}(:),edges);
    
    XLIM = edges([1 end]);
    YLIM = max(cnt) * [-0.1 1.1];
    
    plot(XLIM,[0 0],'k');
    plot([0 0],YLIM,'k');
    
    plot(ctrs,cnt(1:end-1),'b');
    
    set(ax,'XLim',XLIM,'YLim',YLIM);
    if iSub==nSub;
        xlabel('Information Significance (sigmas)');
    end
end
