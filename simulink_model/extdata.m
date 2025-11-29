% === GET SIGNALS FROM logsout ===
ia_sig    = logsout{7}.Values;     % phase A current
ib_sig    = logsout{2}.Values;     % phase B current
ic_sig    = logsout{3}.Values;     % phase C current
speed_sig = logsout{6}.Values;     % rotor speed (rad/s)
torq_sig  = logsout{33}.Values;    % electromagnetic torque (1001 samples)
vab_sig   = logsout{13}.Values;    % line voltage

% === Use ia time vector as the master time ===
t = ia_sig.Time;

% === Resample torque to match 60643 samples ===
torq_resampled = torq_sig.resample(t);

% === Build synchronized dataset ===
data = [t, ...
        ia_sig.Data, ...
        ib_sig.Data, ...
        ic_sig.Data, ...
        speed_sig.Data, ...
        torq_resampled.Data, ...
        vab_sig.Data];

% === CSV header ===
header = {'time','ia','ib','ic','speed','torque','vab'};

% === Write to CSV ===
filename = fullfile(pwd, 'sim_data_healthy.csv');

fid = fopen(filename,'w');
if fid == -1
    error('Failed to open file: %s', filename);
end


fid = fopen(filename,'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
fclose(fid);

dlmwrite(filename, data, '-append');

disp("Healthy dataset exported successfully!");
