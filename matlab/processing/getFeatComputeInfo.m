function [I,Ierr] = getFeatComputeInfo(featname, subj, modtyp, featversion)

% Input handling ----------------------------------------------------------
if nargin<4; featversion=''; end;

% Load data ---------------------------------------------------------------
feat0 = getFeatFromHDF5(featname, subj, 'interictal', modtyp, featversion);
feat1 = getFeatFromHDF5(featname, subj, 'preictal', modtyp, featversion);

% Compute information contained within each element of feature vector
[I,Ierr] = computeFeatInfo(feat0, feat1);

end