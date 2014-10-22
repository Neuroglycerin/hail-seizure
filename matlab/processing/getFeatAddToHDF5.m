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

featM = computeFeatM(featfunc, subj, ictyp, modtyp, nPrt, inparams);

func_str = func2str(featfunc);
if nPrt~=1
    func_str = [num2str(nPrt) func_str];
end

addToHDF5(featM, subj, ictyp, func_str, modtyp, inparams);

end