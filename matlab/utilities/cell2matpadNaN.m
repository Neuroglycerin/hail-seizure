function M = cell2matpadNaN(C)
% Turn a cell array into a matrix with padding of NaNs to make the dimensions in
% each cell match
    
    N = max(ndims(C), max(cellfun('ndims',C)));
    paddingdims = 1:N;
    
    % Cells get merged together along every non-singleton dimension of the
    % cell matrix. If there is only one non-singleton dimension, there is
    % catenation along that dimension, and we don't need to worry about the
    % length of cell contents in that dim.
    if sum(size(C)~=1)==1
        paddingdims = setdiff(paddingdims, find(size(C)~=1));
    end
    
    if isempty(paddingdims)
        M = cell2mat(C);
        return;
    end
    
    maxsizes = nan(1,length(paddingdims));
    for idim = 1:length(paddingdims)
        dimsize = cellfun('size',C,paddingdims(idim));
        maxsizes(idim) = max(dimsize(:));
    end
    
    % Create an anonymous function
    fcn = @(x) pad_nan(x,paddingdims,maxsizes);
    M = cellfun(fcn,C,'UniformOutput',false); % Pad each cell with NaNs
    M = cell2mat(M);                          % collapse down to matrix
    
end

function b = pad_nan(a,paddingdims,maxsizes)
    asize=size(a);
    targetsize = ones(1,max(paddingdims));
    targetsize(1:length(asize)) = asize;
    for idim=1:length(paddingdims)
        dim = paddingdims(idim);
        targetsize(dim) = max(maxsizes(idim),targetsize(dim));
    end
    b=nan(targetsize);
    cart = cell(1,length(asize));
    for idim=1:length(asize)
        cart{idim} = 1:asize(idim);
    end
    b(cart{:}) = a;
end