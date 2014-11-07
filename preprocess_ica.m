% cd to the repository, then run this command
% nice -15 matlab13 -nodisplay -nosplash -r "preprocess_ica_all"

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

extramodtyp = 'cln,dwn';
names = subjnames();

for i=1:length(names);
    computeSubjICAW(names{i}, extramodtyp);
end
