% Parse through the interictal and preictal segments file
% to generate and output an index in the format fname: [seq_id, seg_id]
% Where fname is the filename of a segment mat file (without the .mat for
% hashing reasons)
% Seq_id is the id of the that chunk of contiguous segments
% Seg_id is the id of that segment within that Seq_id
% e.g. "Dog_1_interictal_segment_002" : [1 2]
% this means that segment is the second segment of the first sequence
% 
% All this is dumped to a json

subj = subjnames();
ictyp = {'interictal', 'preictal'};
%test segments don't have this sequence data

nSubj = length(subj);
nIctyp = length(ictyp);

for iSubj=1:nSubj
    for iIctyp=1:nIctyp
        fprintf('Parsing %s %s\n',subj{iSubj}, ictyp{iIctyp});
        [fnames, mydir, segIDs] = subjtyp2dirs(subj{iSubj}, ictyp{iIctyp});
        nFle = length(fnames);
        
        % because I didn't want to guarantee ALL sequences of segements
        % were composed of only 6 segments I used the following hack
        % the sequence id will only iterate if the segment index within 
        % a sequence loops back to a lower value again
        % i.e. if the Seg_ids within sequences go 1,2,3,4,5,6,1   
        % then Seq_id will iterate only after Seg_id becomes a lower 
        % value after 6
        
        sequence_id_number = 1; % initial value for seq_id
        old_seq_id = 0; %initial value for seq_id comparison
        for iFle=1:nFle
            dat = loadSegFile(fullfile(mydir,fnames{iFle}));
            
            % .mat part of fname won't hash so trim off last 4 chars 
            len = length(dat.filename); 
            seg_fname = dat.filename(1:len-4); 
            
            new_seq_id = dat.sequence;
            
            % check if new seg_id is lower value than previous
            if (new_seq_id < old_seq_id)
                % if so iterate the seq id
                sequence_id_number = sequence_id_number+1;
            end
            old_seq_id = new_seq_id;
            
            % add data to hash table
            index_mat.(seg_fname) = [sequence_id_number old_seq_id];
        end    
    end
end

%output hash table to json
json.write(index_mat, 'sequence_sequment_id_index.json');