% setup.m - Run this script to set up the project paths (if script in folder "scripts")

% Get the root directory (where setup.m is located)
rootDir = fileparts(mfilename('fullpath'));

% Path to functions folder
functionsPath = fullfile(rootDir, 'functions');

% Check if functions are already in path, if not, add them
if ~contains(path, functionsPath)
    addpath(genpath(functionsPath));
    disp('Setup complete: functions folder added to path.');
else
    disp('Setup already complete: functions folder is in path.');
end
