% SEEDRNG Seeds the MATLAB random number generator, and Shuffle.m
%    Seeds using the clock milliseconds, so seeding is unpredictable
%    and will be different on each run.
%    Outputs the seeds used, incase you need to save them for posterity.

function [rgnseed shuffleseed] = seedrng()

% Seed MATLAB random number generator with clock millisecond time
c = clock;
rgnseed = round(2^20*rem(c(end),1));
rng(rgnseed);

% Use MATLAB random number generator, to make a seed for Shuffle
shuffleseed = randi(2^32,[1 4])-1; % Generate a seed for Shuffle
Shuffle(shuffleseed, 'seed');  % Seed Shuffle

end