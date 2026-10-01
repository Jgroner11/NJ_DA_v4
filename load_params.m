function P = load_params()
%LOAD_PARAMS parameters.yaml, with the chosen session's data_file and maze_png
% lifted to the top level, where every script reads them.
%
% The session is the one the PIPELINE_SESSION environment variable names,
% which pipeline.py sets for each session in turn. Without it -- a script run
% by hand -- it is the first one listed under `sessions`. The two files always
% come from the same entry. The MATLAB twin of paths.load_params; keep the two
% in step.
P = yaml.loadFile('parameters.yaml');
assert(isfield(P, 'sessions') && isstruct(P.sessions), ...
       'no sessions listed in parameters.yaml -- are they all commented out?');
names = string(fieldnames(P.sessions));

name = string(getenv('PIPELINE_SESSION'));
if name == ""
    name = names(1);
    fprintf('PIPELINE_SESSION not set; running %s, the first session in parameters.yaml\n', name);
end
assert(any(names == name), 'no session %s in parameters.yaml; it has %s', ...
       name, strjoin(names, ', '));

P.session = name;
P.data_file = P.sessions.(name).data_file;
P.maze_png = P.sessions.(name).maze_png;
end
