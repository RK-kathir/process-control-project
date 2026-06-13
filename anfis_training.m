
% ════════════════════════════════════════════════════════════════════════
%  anfis_training.m
%  Trains Kp_Data.fis and Ki_Data.fis from the Tuning Bot ANFIS dataset.
%
%  Data source: the Tuning Bot server logs (Disturbance, Kp, Ki) every
%  time a tune_request with a "disturbance" field is sent (see
%  matlab_bridge.py). This script loads that dataset directly from the
%  server (or from a local CSV downloaded by the bridge) and runs the
%  existing ANFIS training pipeline unchanged.
% ════════════════════════════════════════════════════════════════════════
 
clear; clc;
 
% ── CONFIG ────────────────────────────────────────────────────────────
SERVER_URL = 'https://process-control-project.onrender.com';
USE_LOCAL_CSV = false;          % true → read local CSV instead of server
LOCAL_CSV_PATH = 'anfis_training_data.csv';
 
 
% ── 0. Load training data ────────────────────────────────────────────
fprintf('--- Loading ANFIS training data ---\n');
 
if USE_LOCAL_CSV
    fprintf('Reading local file: %s\n', LOCAL_CSV_PATH);
    T = readtable(LOCAL_CSV_PATH);
else
    csv_url = [SERVER_URL '/api/anfis-data.csv'];
    fprintf('Fetching: %s\n', csv_url);
    T = readtable(csv_url);
end
 
if height(T) < 5
    error(['Not enough training data (%d rows). Run a disturbance sweep ' ...
           'first — see matlab_bridge.py, Example 3.'], height(T));
end
 
D_values  = T.disturbance;
Kp_values = T.kp;
Ki_values = T.ki;
 
fprintf('Loaded %d training points.\n', height(T));
fprintf('  Disturbance range: [%.3f, %.3f]\n', min(D_values), max(D_values));
fprintf('  Kp range:          [%.4f, %.4f]\n', min(Kp_values), max(Kp_values));
fprintf('  Ki range:          [%.4f, %.4f]\n', min(Ki_values), max(Ki_values));
 
 
% ── 1. ANFIS Training: Converting Data to .fis Files ─────────────────
% (Original pipeline — unchanged)
fprintf('\n--- Starting ANFIS Training ---\n');
 
% 1. Format the data into vertical columns
InputData = D_values;     % Disturbance column (already vertical from readtable)
OutputKp  = Kp_values;     % Kp values
OutputKi  = Ki_values;     % Ki values
 
% Combine inputs and outputs into training matrices
TrainData_Kp = [InputData, OutputKp];
TrainData_Ki = [InputData, OutputKi];
 
% 2. Generate the Initial Fuzzy Structures
% This looks at your data and creates a default set of membership functions
fisOptions = genfisOptions('GridPartition');
initial_fis_Kp = genfis(InputData, OutputKp, fisOptions);
initial_fis_Ki = genfis(InputData, OutputKi, fisOptions);
 
% 3. Train the ANFIS Models
% The 'anfis' command runs the neural network training to adjust the rules
% The [100 0 0.01 0.9 1.1] are standard training options (100 epochs/cycles)
disp('Training Kp ANFIS...');
trained_fis_Kp = anfis(TrainData_Kp, initial_fis_Kp, [100 0 0.01 0.9 1.1]);
disp('Training Ki ANFIS...');
trained_fis_Ki = anfis(TrainData_Ki, initial_fis_Ki, [100 0 0.01 0.9 1.1]);
 
% 4. Save the trained models as .fis files!
writeFIS(trained_fis_Kp, 'Kp_Data');
writeFIS(trained_fis_Ki, 'Ki_Data');
fprintf('\nSuccess: Kp_Data.fis and Ki_Data.fis have been generated!\n');
 
 
% ── 2. Quick visual check of the learned surfaces ────────────────────
figure('Name','ANFIS Training Result');
 
subplot(1,2,1);
plot(D_values, Kp_values, 'o', 'Color', [0 0.78 1]); hold on;
d_fine = linspace(min(D_values), max(D_values), 200)';
plot(d_fine, evalfis(trained_fis_Kp, d_fine), '-', 'Color', [0.49 0.55 1], 'LineWidth', 2);
xlabel('Disturbance D'); ylabel('Kp'); title('Kp(D) — ANFIS fit');
legend('Training data','ANFIS surface','Location','best'); grid on;
 
subplot(1,2,2);
plot(D_values, Ki_values, 'o', 'Color', [1 0.3 0.42]); hold on;
plot(d_fine, evalfis(trained_fis_Ki, d_fine), '-', 'Color', [0.18 0.83 0.63], 'LineWidth', 2);
xlabel('Disturbance D'); ylabel('Ki'); title('Ki(D) — ANFIS fit');
legend('Training data','ANFIS surface','Location','best'); grid on;
 
 
% ── 3. Point Simulink ANFIS blocks at the new .fis files ─────────────
fprintf('\nNext steps:\n');
fprintf('  1. Open your Simulink model.\n');
fprintf('  2. Double-click the Kp ANFIS block (Fuzzy Logic Controller).\n');
fprintf('  3. Set "FIS" source = File, point to Kp_Data.fis\n');
fprintf('  4. Repeat for Ki ANFIS block -> Ki_Data.fis\n');
fprintf('  5. Re-run the simulation. New telemetry/tune_request calls will\n');
fprintf('     add more points to the Tuning Bot dataset for the next retrain.\n');
 
% ── Optional: reset the server dataset before the NEXT sweep ─────────
% webwrite([SERVER_URL '/api/anfis-reset'], struct());
