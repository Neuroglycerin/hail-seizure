function param = parammerge(defaultparam, inputparam)

param = defaultparam;
fnms = intersect(fieldnames(param), fieldnames(inputparam));

for iFld=1:length(fnms)
    param.(fnms{iFld}) = inputparam.(fnms{iFld});
end

end