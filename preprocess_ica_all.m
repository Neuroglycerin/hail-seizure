% cd to the repository, then run this command
% matlab13 -nodisplay -nosplash -r "preprocess_ica_all"

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

%names = subjnames();
names = {'Patient_1'};
for i=1:length(names);
    preprocess_ica(names{i});
end
