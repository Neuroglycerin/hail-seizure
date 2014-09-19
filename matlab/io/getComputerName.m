%GETCOMPUTERNAME Returns the name of the computer (hostname)
function name = getComputerName()

[ret, name] = system('hostname');   

if ret == 0
    return;
end

if ispc
    name = getenv('COMPUTERNAME');
else      
    name = getenv('HOSTNAME');      
end

end