% ICRA Goal Analysis
addpath('/home/keshav/repos/libraries/PlotPub/')
addpath('/home/keshav/repos/libraries/PlotPub/lib/')
%% Load in data
cd '/home/keshav/ctm2-stable-baselines/saved_results/icra_experiments'
experiments = ["cras_exp_1_new", "cras_exp_2", "cras_exp_3", "cras_exp_4_new", ...
    "cras_exp_5", "cras_exp_6"];
for i=1:length(experiments)
    temp_table = readtable(strcat('data/', experiments(i), '_joint_error_analysis.csv'));
    temp_table.experiment = repmat(experiments(i), 1000, 1);
    if i == 1
        exp_table = temp_table;
    else
        exp_table = [exp_table; temp_table];
    end
end
% Compute initial goal distances and final errors
[initial_errors, final_errors] = computeErrors(exp_table);
exp_table.initial_errors = initial_errors;
exp_table.final_errors = final_errors;

%% 1. q_desired vs. error plots
% Does desired joint position affect final error?
% Extension
BetaDesiredErrorRegression(exp_table, experiments)
%% Rotation
exp_idx = exp_table.experiment == experiments(6);
figure;
alpha_desired = [exp_table.alpha_desired_1(exp_idx); exp_table.alpha_desired_2(exp_idx); exp_table.alpha_desired_3(exp_idx)];
polarscatter(alpha_desired, repmat(exp_table.final_errors(exp_idx)*1000, 3, 1))
pax = gca;
pax.ThetaLim = [-180 180];
pax.ThetaZeroLocation = 'top';
AlphaDesiredPolarPlot(exp_table, experiments, 7)
%% 2. Delta q (desired_q - starting_q vs error)
exp_idx = exp_table.experiment == experiments(6);
scatter(vecnorm(exp_table.B_desired_1(exp_idx) - exp_table.B_starting_1(exp_idx), 2, 2), exp_table.final_errors(exp_idx))
hold on
scatter(vecnorm(exp_table.B_desired_2(exp_idx) - exp_table.B_starting_2(exp_idx), 2, 2), exp_table.final_errors(exp_idx))
hold on
scatter(vecnorm(exp_table.B_desired_3(exp_idx) - exp_table.B_starting_3(exp_idx), 2, 2), exp_table.final_errors(exp_idx))

% Rotation
polarscatter(vecnorm(exp_table.alpha_desired_1(exp_idx) - exp_table.alpha_starting_1(exp_idx), 2, 2), exp_table.final_errors(exp_idx))
hold on
polarscatter(vecnorm(exp_table.alpha_desired_2(exp_idx) - exp_table.alpha_starting_2(exp_idx), 2, 2), exp_table.final_errors(exp_idx))
hold on
polarscatter(vecnorm(exp_table.alpha_desired_3(exp_idx) - exp_table.alpha_starting_3(exp_idx), 2, 2), exp_table.final_errors(exp_idx))

%% Plot achieved goals with errors
exp_id = 'cras_exp_6';
tol = 0.005;
ag_x = exp_table.achieved_goal_x(exp_table.experiment == exp_id & exp_table.final_errors > tol) * 1000;
ag_y = exp_table.achieved_goal_y(exp_table.experiment == exp_id & exp_table.final_errors > tol) * 1000;
ag_z = exp_table.achieved_goal_z(exp_table.experiment == exp_id & exp_table.final_errors > tol) * 1000;

final_errors = exp_table.final_errors(exp_table.experiment == exp_id & exp_table.final_errors > tol) * 1000;
scatter3(ag_x, ag_y, ag_z, 10, final_errors);
%% Functions
function AlphaDesiredPolarPlot(exp_table, experiments, k)
for i=1:length(experiments)
    exp_idx = exp_table.experiment == experiments(i);
    figure;
    alpha_desired = [exp_table.alpha_desired_1(exp_idx); exp_table.alpha_desired_2(exp_idx); exp_table.alpha_desired_3(exp_idx)];
    [u,v] = pol2cart(alpha_desired, repmat(exp_table.final_errors(exp_idx)*1000, 3, 1));
    [idx,C] = kmeans([u, v], k);
    [theta,rho] = cart2pol(C(:,1), C(:,2));
    polarscatter(theta, rho, groupcounts(idx), 'filled','MarkerFaceAlpha',.5)
    pax = gca;
    pax.ThetaLim = [-180 180];
    pax.ThetaZeroLocation = 'top';
    title(experiments(i),'interpreter', 'none')

end
end


function BetaDesiredErrorRegression(exp_table, experiments)
for tube=1:3
    figure;
    for i=1:length(experiments)
        exp_idx = exp_table.experiment == experiments(i);
        switch tube
            case 1
                B_desired = exp_table.B_desired_1(exp_idx);
            case 2
                B_desired = exp_table.B_desired_2(exp_idx);
            case 3
                B_desired = exp_table.B_desired_3(exp_idx);
        end
        p = polyfit(B_desired * 1000, exp_table.final_errors(exp_idx) * 1000, 1)
        f = polyval(p, B_desired * 1000);
        plot(B_desired*1000, f)
        hold on
    end
    opt = [];
    opt.XLabel = ['Desired retraction tube ', int2str(tube), ' (mm)']; % xlabel
    %opt.XLim = [0, 140];
    opt.YLabel = 'final error (mm)'; %ylabel
    %opt.YLim = [0, 20];
    opt.FontName = 'Times';
    opt.FontSize = 14;
    %opt.Markers = {'o', ''};
    %opt.LineStyle = {'None', '-'}; % line width
    legend(experiments, 'Interpreter', 'none')
    % apply the settings
    setPlotProp(opt);
end
end

function [initial_errors, final_errors] = computeErrors(experiment)
initial_errors = vecnorm([experiment.desired_goal_x, ...
    experiment.desired_goal_y, experiment.desired_goal_z] - ...
    [experiment.starting_position_x, experiment.starting_position_y, ...
    experiment.starting_position_z], 2, 2);
final_errors = vecnorm([experiment.desired_goal_x, ...
    experiment.desired_goal_y, experiment.desired_goal_z] - ...
    [experiment.achieved_goal_x, experiment.achieved_goal_y, ...
    experiment.achieved_goal_z], 2, 2);
end