% Get feature then add to appropriate archive
% Generate feature vector and append to HDF5 using get HDF5
% Inputs : str subj         - which subject e.g. Dog_[1-5], Patient_[1-2]
%        : str typ          - which datasegment type e.g. preictal, ictal, test
%        : handle func      - function handle to feature generation func
%                             must output featM as first output
% Outputs: void 

function getFeatAddToHDF5(subj, typ, func)

    addToHDF5(subj, typ, func2str(func), func(subj, typ));
