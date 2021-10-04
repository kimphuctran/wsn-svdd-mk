clear all;close all;clc;
addpath(genpath(pwd));

%% Data Reading
clear all;close all;clc;

% Read file
fileID=fopen('data.txt');
formatSpec='2004-%f-%f %f:%f:%f %f %f %f %f %f %f';
ibrlData=textscan(fileID,formatSpec);
fclose(fileID);

% Extract all data
month=ibrlData{1};
date=ibrlData{2};

hour=ibrlData{3};
minute=ibrlData{4};
second=ibrlData{5};
time=hour*3600+minute*60+second;

moteid=ibrlData{7};
temperature=ibrlData{8};
humidity=ibrlData{9};

save ibrl_data month date time moteid temperature humidity;

%% Sensor Data
clear all;close all;clc;
load ibrl_data;

% Extract data given time window
ibrlData=[month date time moteid temperature humidity];
ibrlData(isnan(ibrlData(:,5)),:)=[];
ibrlData(isnan(ibrlData(:,6)),:)=[];
ibrlData(ibrlData(:,1)~=3,:)=[];  
ibrlData(ibrlData(:,2)>10,:)=[];
ibrlData(:,[1 2 3])=[]; 

% Extract mote data (only sensor 1, 2, 33, 35, 37 are considered)
moteIDs=[1 2 33 35 37];
trainData=[];
for i=moteIDs
    trainData=[trainData;ibrlData(ibrlData(:,1)==i,1:3)];
end

% Visualization
colorList=rand(length(moteIDs),3);
for i=1:length(moteIDs)
    figure(1);
    subplot(211);
    plot(ibrlData(ibrlData(:,1)==i,2),...
         '-','Color',colorList(i,:));
    hold on;
    subplot(212);
    plot(ibrlData(ibrlData(:,1)==i,3),...
         '-','Color',colorList(i,:));
    hold on;
    figure(2);
    plot(trainData(trainData(:,1)==moteIDs(i),2),...
         trainData(trainData(:,1)==moteIDs(i),3),...
         '*','Color',colorList(i,:));
    hold on;
end

save train_data trainData;

%% Training procedure
clear all;close all;clc;

clear all;close all;clc;
load train_data;
trainData=consolidator(trainData,[],@mean,2e-2);

[suspicious_index,lof]=local_outlier_factor(trainData(:,2:3),50);

[~,I]=sort(suspicious_index,1,'descend');
negativeIndex=suspicious_index(1:ceil(1e-2*length(I)));
positiveIndex=suspicious_index(ceil(1e-2*length(I))+1:end);

positiveData=trainData(positiveIndex,2:3);
negativeData=trainData(negativeIndex,2:3);

figure(1);clf;
scatter(positiveData(:,1),positiveData(:,2),'bo');
hold on;
scatter(negativeData(:,1),negativeData(:,2),'rx');

%
normalizedData=bsxfun(@rdivide,...
    positiveData-repmat(min(positiveData),size(positiveData,1),1),...
    max(positiveData)-min(positiveData));
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

ocSVM.normalizeLB=min(positiveData);
ocSVM.normalizeUB=max(positiveData); 
ocSVM.gamma=2e-2; % control limit adjustment

% Bayesian optimization
algorithmList={NLOPT_GN_DIRECT NLOPT_GN_DIRECT_L NLOPT_GN_DIRECT_L_RAND...
               NLOPT_GN_CRS2_LM...
               NLOPT_GN_ESCH...
               NLOPT_GN_ISRES};
opt.algorithm=algorithmList{2};
opt.xtol_abs=[1e-3;1e-3];
opt.ftol_abs=1e-4;
opt.maxeval=1e2;
opt.max_objective=...
    @(x) svdd_gmean(ocSVM,trainData,trainLabel,positiveData,negativeData,x);
opt.lower_bounds=[5e-3;1/128];
opt.upper_bounds=[1;1];
opt.verbose=1;
opt.initial_step=[1e-2;1e-2];
xopt=nlopt_optimize_mex(opt,[1 .5]);

% Optimal hyperparameters
ocSVM.C=[xopt(1) 0];
ocSVM.sigma=xopt(2);
ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);

save ocsvm ocSVM;

% Boundary visualization
testData=repmat(ocSVM.normalizeLB-10,1e6,1)+...
    bsxfun(@times,rand(1e6,2),(ocSVM.normalizeUB-ocSVM.normalizeLB+20));
[predictLabel,boundaryLabel]=svdd_classify(ocSVM,testData);

figure(2);clf;
plot(positiveData(:,1),positiveData(:,2),'r*');hold on;
plot(negativeData(:,1),negativeData(:,2),'b*');hold on;
plot(testData(boundaryLabel==1,1),testData(boundaryLabel==1,2),'go','linewidth',2);

%% Data validation
% Labelling
clear all;close all;clc;
load ibrl_data;

ibrlData=[month date time moteid temperature humidity];
ibrlData(:,[1 2 3])=[]; 

for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
end

I{1}=[1 2411 2417 2501 2504 2509 2520 2521 2534 2.26e4:2.38e4];
J{1}=[4e4:4.3e4];

I{2}=[1 276 4602 4611 4613 4628 4630 1.88e4:1.98e4];
J{2}=[4.25e4:4.65e4];

I{33}=[1.42e4:1.48e4];
J{33}=[3.3e4:3.48e4];

I{35}=[4.5e4:4.55e4];
J{35}=[3.5e4:3.6e4];

I{37}=unique(max(2.046e4,min(2.36e4,find(moteData{37}(:,1)>37))));
I{37}(1)=[];I{37}(end)=[];
J{37}=[4.7e4:4.74e4];

negativeData=[];
positiveData=[];
for i=[1 2 33 35 37]
    negativeData=[negativeData;moteData{i}(I{i},:)];
    positiveData=[positiveData;moteData{i}(J{i},:)];
    figure(i)
    plot(moteData{i},'b-');hold on;
    plot(I{i},moteData{i}(I{i},:),'ro');hold on;
    plot(J{i},moteData{i}(J{i},:),'go');
end

save test_data positiveData negativeData;

% Validation
load ocsvm;
testData=[positiveData;negativeData];
trueLabel=[ones(size(positiveData,1),1);-1*ones(size(negativeData,1),1)];
[predictLabel,boundaryLabel]=svdd_classify(ocSVM,testData);

figure;clf;
plot(positiveData(:,1),positiveData(:,2),'b*');hold on;
plot(negativeData(:,1),negativeData(:,2),'r*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);

DR=length(find(predictLabel==-1 & trueLabel==-1))/size(negativeData,1);
FNR=1-DR;
FPR=length(find(predictLabel==-1 & trueLabel==1))/size(positiveData,1);
