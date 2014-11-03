% Get feature then add to appropriate archive
% Generate feature vector and append to HDF5 using get HDF5
% Inputs : handle featfunc  - function handle to feature generation featfunc
%                             must output featM as first output
%        : str subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str ictyp        - which datasegment type e.g. preictal, ictal, test
%        : str modtyp       - which preprocessing model to apply
%        : int nPrt         - number of splits of the data segments
%        : [struct inparams]- input parameters for the featfunc
%        
% Outputs: void

function getFeatAddToHDF5(featfunc, subj, ictyp, modtyp, nPrt, inparams)

% Default inputs ----------------------------------------------------------
if nargin<5
    nPrt = 1;
end
if nargin<6
    inparams = struct([]);
end
% Input handling ----------------------------------------------------------
if ischar(featfunc)
    featfunc = str2func(featfunc);
end

% Compute the
[featM,outparams] = computeFeatM(featfunc, subj, ictyp, modtyp, nPrt, inparams);

% Get our function name from its handle
func_str = func2str(featfunc);
if nPrt~=1
    func_str = [num2str(nPrt) func_str];
end

% Check if multiple feature names are present
if ~isfield(outparams,'featnames') || isempty(outparams.featnames)
    dim = [];
else
    % If it is, find which dimension is across multiple subfeatures
    siz = size(outparams.featnames);
    dim = find(siz~=1);
end
if iscell(featM{1,2})
    if isempty(dim)
        outparams.featnames = cell(numel(featM{1,2}),1);
        for iSbf=1:numel(featM{1,2})
            outparams.featnames{iSbf} = num2str(iSbf);
        end
    end
    if numel(outparams.featnames)~=numel(featM{1,2})
        error('Inconsistent subfeature lengths');
    end
    for iSbf=1:numel(featM{1,2})
        subfeatM = cell(size(featM));
        subfeatM(:,1) = featM(:,1);
        for iFle=1:size(featM,1)
            subfeatM{iFle,2} = featM{iFle,2}{iSbf};
        end
        addToHDF5(subfeatM, subj, ictyp, [func_str '-' outparams.featnames{iSbf}], modtyp, inparams);
    end
elseif isempty(dim)
    % Only one feature, so write to regular name
    addToHDF5(featM, subj, ictyp, func_str, modtyp, inparams);
else
    % Make a cell to take everything from the array
    allcell = cell(size(siz));
    for iDim=1:length(siz)
        allcell{iDim} = ':';
    end
    % Separate out the subfeatures and save in separate files
    for iSbf=1:length(outparams.featnames);
        pp = allcell;
        pp{dim} = iSbf;
        subfeatM = cell(size(featM));
        subfeatM(:,1) = featM(:,1);
        for iFle=1:size(featM,1)
            subfeatM{iFle,2} = featM{iFle,2}(pp{:});
        end
        addToHDF5(subfeatM, subj, ictyp, [func_str '-' outparams.featnames{iSbf}], modtyp, inparams);
    end
end

end