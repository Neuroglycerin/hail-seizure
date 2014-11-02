% cd to the repository, then run this command
% nice -15 matlab13 -nodisplay -nosplash -r "preprocess_ica_all"

% Recursively add all of matlab folder to matlab path
addpath(genpath('matlab'));

names = subjnames();

for i=1:length(names);
    fprintf('%s: Computing CSP weights for %s\n',datestr(now,30),names{i});
    saveCSPweights(names{i});
end