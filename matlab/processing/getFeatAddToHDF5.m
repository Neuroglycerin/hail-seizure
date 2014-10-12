% Get feature then add to appropriate archive
% Generate feature vector and append to HDF5 using get HDF5
% Inputs : handle featfunc  - function handle to feature generation featfunc
%                             must output featM as first output
%        : str subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str ictyp        - which datasegment type e.g. preictal, ictal, test
%        : str modtyp       - which preprocessing model to apply
%        : [struct inparams]- input parameters for the featfunc
%        
% Outputs: void

function getFeatAddToHDF5(featfunc, subj, ictyp, modtyp, inparams)

% Default inputs ----------------------------------------------------------
if nargin<5
    inparams = struct([]);
end
% Input handling ----------------------------------------------------------
if ischar(featfunc)
    featfunc = str2func(featfunc);
end

featM = computeFeatM(featfunc, subj, ictyp, modtyp, inparams);

addToHDF5(featM, subj, ictyp, func2str(featfunc), modtyp, inparams);

end