% Write data to dataset in H5 file.
% - If dataset does not exist, create it
% - If dataset already exists, overwrite it
function h5writePlus(h5fnme, dataset, data)

% fprintf('Going to write to %s the dataset %s\n',h5fnme,dataset);
% return;

% h5create will throw an error if the dataset already exists
try
    h5create(h5fnme, dataset, ...
        size(data), ...
        'Datatype', class(data));
    h5write(h5fnme, dataset, data);
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:imagesci:h5create:datasetAlreadyExists')
        rethrow(ME);
    end
    % if the dataset does exist and the try throws an exception then we
    % can just call h5write by itself to overwrite the dataset
    try
        h5write(h5fnme, dataset, data);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:imagesci:h5write:fullDatasetDataMismatch')
            throw(ME);
        else
            rethrow(ME);
        end
    end
end

end
