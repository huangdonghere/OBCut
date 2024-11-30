function demo_OBCut()

clear;
close all;
clc;

dataName = 'yale';
% dataName = 'Mpeg7';
load(['data_',dataName,'.mat'],'fea','gt'); 

k = numel(unique(gt)); % The number of clusters
M = 100; %The number of anchors
lambda = 1; % The trade-off parameter

tic;
disp("Running OBCut...")
Label = OBCut(fea,k,M,lambda);
disp("Done.");
toc;

disp('The NMI score on this dataset:');
scores = NMImax(Label,gt);
disp(['NMI = ',num2str(scores)]);
