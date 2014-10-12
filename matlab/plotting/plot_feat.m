modtyp = 'raw';
featname = 'feat_psd_logf';
subj = 'Dog_1';

% Load data
feat0 = getFeatFromHDF5(featname, subj, 'interictal', modtyp);
feat1 = getFeatFromHDF5(featname, subj, 'preictal', modtyp);

% Reshape so we lose any dimensions higher than 3
feat0 = reshape(feat0,size(feat0,1),size(feat0,2),[]);
feat1 = reshape(feat1,size(feat1,1),size(feat1,2),[]);

% Check sizing
siz0 = size(feat0);
siz1 = size(feat1);
if ~isequal(siz0(2:end),siz1(2:end)); error('unbalanced sizes'); end;
sizr = [max(siz0(1),siz1(1)) siz0(2) siz0(3)];

% if length(siz)>3; error('too many dims'); end;

% mu0 = mean(feat0,1);
% se0 = std(feat0,1)/sqrt(size(feat0,1));
% mu1 = mean(feat1,1);
% se1 = std(feat1,1)/sqrt(size(feat1,1));

%%
% figure;
for iD3=1:sizr(3)
    % subplot(sizr(3),1,iD3);
    figure;
    % Merge pre and inter together
    ff = nan([sizr(1) 3*sizr(2)]);
    ff(1:siz0(1),1:3:end) = feat0(:,:,iD3);
    ff(1:siz1(1),2:3:end) = feat1(:,:,iD3);
    boxplot(ff,'plotstyle','compact','symbol','.','colors','brw');
end