%GETCOMPUTERNAME Returns the name of the computer (hostname)
function name = getComputerName()

[ret, name] = system('hostname');   

name = strtrim(name);

if ret == 0
    return;
end

if ispc
    name = getenv('COMPUTERNAME');
else      
    name = getenv('HOSTNAME');      
end

name = strtrim(name);

end
