%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                      %
% This is a demo for the OBCut algorithm, which is proposed in the     %
% following paper:                                                     %
%                                                                      %
% Si-Guo Fang, Dong Huang, Chang-Dong Wang, Jian-Huang Lai.            %
% One-step Bipartite Graph Cut: A Normalized Formulation and Its       %
% Application to Scalable Subspace Clustering.                         %                                      %
% Neural Networks, accepted, 2026.                                     %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_OBCut()

clear;
close all;
clc;

% dataName = 'yale';
dataName = 'Mpeg7';
load(['data_',dataName,'.mat'],'fea','gt'); 

k = numel(unique(gt)); % The number of clusters
M = 100; % The number of anchors
lambda = 1; % The trade-off parameter

tic;
disp("Running OBCut...")
Label = OBCut(fea,k,M,lambda);
disp("Done.");
toc;

% tic;
% % Run the paralleled version.
% % Faster for large-scale datasets.
% disp("Running paralleled OBCut...")
% Label = OBCut_par(fea,k,M,lambda);
% disp("Done.");
% toc;

disp('The NMI score on this dataset:');
scores = NMImax(Label,gt);
disp(['NMI = ',num2str(scores)]);
