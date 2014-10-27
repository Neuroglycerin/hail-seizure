function [I,Ierr] = getFeatComputeInfo(featname, subj, modtyp, featversion)

% Input handling ----------------------------------------------------------
if nargin<4; featversion=''; end;

% Load data ---------------------------------------------------------------
feat0 = getFeatFromHDF5(featname, subj, 'interictal', modtyp, featversion);
feat1 = getFeatFromHDF5(featname, subj, 'preictal', modtyp, featversion);
% Load pseudo training data if we can
try
    feat0p = getFeatFromHDF5(featname, subj, 'pseudointerictal', modtyp, featversion);
    feat1p = getFeatFromHDF5(featname, subj, 'pseudopreictal', modtyp, featversion);
    % Merge real and pseudo training data
    if iscell(feat0)
        % Cell merge
        feat0 = {feat0; feat0p};
        feat1 = {feat1; feat1p};
    else
        % Matrix merge
        feat0 = [feat0; feat0p];
        feat1 = [feat1; feat1p];
    end
    % Note we have pseudo
    haspseudo = true;
catch ME
    if ~strcmp(ME.identifier,'getFeatFromHDF5:missingIctal')
        % Carry on without pseudo training data
        rethrow(ME);
    end
    % Note we don't have pseudo
    haspseudo = false;
end

% Compute information contained within each element of feature vector
[I,Ierr] = computeFeatInfo(feat0, feat1, haspseudo);

end