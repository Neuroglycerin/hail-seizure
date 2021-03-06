% func should be a handle to a feature function which accepts two inputs
% the datastructure, Dat, and a structure of input parameters

function [featM,outparams] = computeFeatM(featfunc, subj, ictyp, modtyp, nPrt, inparams)

% Input handling ----------------------------------------------------------
if nargin<6
    inparams = struct([]); % Empty struct
end
if ischar(featfunc)
    featfunc = str2func(featfunc);
end

% Setup -------------------------------------------------------------------
% Get the preprocessing function to use
ppfunc = getPreprocFunc(modtyp, subj);

% Check if we are generating pseudo training data
if strcmp('pseudopreictal',ictyp)
    ictyp = 'preictal';
    ispseudo = true;
elseif strcmp('pseudointerictal',ictyp)
    ictyp = 'interictal';
    ispseudo = true;
else
    ispseudo = false;
end

% Get a list of files
[fnames, mydir, segIDs] = subjtyp2dirs(subj, ictyp, modtyp);
nFle = length(fnames);

% Parallelised feature computation ----------------------------------------
% Loop over each segment, computing feature and adding to matrix

featM = cell(nFle, 2);
fleIsDoable = false(nFle, 1);
alloutparams = cell(nFle, 1);

parfor iFle=1:nFle
    % Load this segment
    Dat = loadSegFile(fullfile(mydir,fnames{iFle}));
    % Note how many parts there should be
    mynPrt = nPrt;
    % Handle pseudo training datasets
    if ispseudo
        if iFle~=nFle;
            % Load the next file to merge with it
            Dat2 = loadSegFile(fullfile(mydir,fnames{iFle+1}));
            % Check consequtive files
            if Dat2.segID ~= Dat.segID+1
                error('Need to go through segIDs sequentially!');
            end
        else
            % If this is the last file, can't merge with the next anymore
            % Alias to Dat so we have something to compare sequence with
            Dat2 = Dat;
        end
        % Will offset by half a segment-part so all segparts are unique
        k = round(size(Dat.data,2)/nPrt/2);
        % Check if sequential recordings
        if Dat2.sequence == Dat.sequence+1;
            % Should merge sequential recordings together
            % Check sampling rate is consistent
            if abs(Dat2.fs - Dat.fs) > 1e-8;
                error('Inconsistent segment sampling rates');
            end
            % Merge data segments together
            Dat.data = cat(2, Dat.data(:,k+1:end), Dat2.data(:,1:k));
        elseif nPrt==1
            % No need to do this one; already did as much as we can
            continue;
        else
            % Crop off the start and end half segments
            Dat.data = Dat.data(:,k+1:end-k+1);
            % We will be able to get one fewer segpart out of this segment
            mynPrt = nPrt-1;
        end
        % Modify other structure attributes
        Dat.sequence = Dat.sequence + 0.5;
        Dat.segID = Dat.sequence + 0.5;
        % Virtual filename
        myfname = strrep(fnames{iFle},ictyp,['pseudo' ictyp]);
    else
        % Regular filename
        myfname = fnames{iFle};
    end
    
    % Apply the preprocessing model
    Dat = ppfunc(Dat);
    % The preprocessing model might change the datalength
    prtlen = floor(size(Dat.data,2)/mynPrt);
    
    % Compute the feature for each part of the recording
    % Do the first part now
    [part_vec,alloutparams{iFle}] = featfunc(splitPart(Dat,1,prtlen), inparams);
    % Check size of the part
    part_siz = size(part_vec);
    if iscell(part_vec)
        feat_vec = cell(part_siz);
        sbf_siz = cell(part_siz);
        maxndim = 0;
        for iSbf=1:numel(feat_vec)
            sbf_siz{iSbf} = size(part_vec{iSbf});
            feat_vec{iSbf} = nan([sbf_siz{iSbf}(1)*mynPrt sbf_siz{iSbf}(2:end)]);
            maxndim = max(maxndim,length(sbf_siz{iSbf}));
        end
        % Make a cell to extract every dimension
        cln = cell(maxndim-1,1);
        for iCln=1:length(cln)
            cln{iCln} = ':';
        end
        for iSbf=1:numel(feat_vec)
            % Add the first part now
            feat_vec{iSbf}(1:sbf_siz{iSbf}(1), cln{:}) = part_vec{iSbf};
        end
        % Do the remaining parts
        for iPrt=2:mynPrt
            prt_cell = featfunc(splitPart(Dat,iPrt,prtlen), inparams);
            for iSbf=1:numel(feat_vec)
                feat_vec{iSbf}(sbf_siz{iSbf}(1)*(iPrt-1)+(1:sbf_siz{iSbf}(1)), cln{:}) = ...
                    prt_cell{iSbf};
            end
        end
    else
        if part_siz(1)>1;
            warning('Features should have singleton first dimension');
        end;
        % Make holding variable for all the parts
        feat_siz = [part_siz(1)*mynPrt part_siz(2:end)];
        feat_vec = nan(feat_siz);
        maxndim = length(feat_siz);
        % Make a cell to extract every dimension
        cln = cell(maxndim-1,1);
        for iCln=1:length(cln)
            cln{iCln} = ':';
        end
        % Add the first part now
        feat_vec(1:part_siz(1), cln{:}) = part_vec;
        % Do the remaining parts
        for iPrt=2:mynPrt
            feat_vec(part_siz(1)*(iPrt-1)+(1:part_siz(1)), cln{:}) = ...
                featfunc(splitPart(Dat,iPrt,prtlen), inparams);
        end
    end
    % Remove cln_ from beginning of filename
    if strncmp(myfname,'cln_',4)
        myfname = myfname(5:end);
    end
    % Add to the cell
    featM(iFle, :) = {myfname, feat_vec};
    % Note that we managed to do this file
    fleIsDoable(iFle) = true;
end

% Cut out the empty data
featM = featM(fleIsDoable,:);

% Assume always the same output parameters
outparams = alloutparams{1};

end

% Small subfunction for neatly spltting data into parts
function Dat = splitPart(Dat,iPrt,prtlen)

    % Cut down data size
    Dat.data = Dat.data(:,(iPrt-1)*prtlen+1:iPrt*prtlen);
    
end
