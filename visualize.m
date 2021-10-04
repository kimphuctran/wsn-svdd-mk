% Storage
trainData=trainData.*(ocSVM.normalizeUB-ocSVM.normalizeLB)+ocSVM.normalizeLB;
positiveData=consolidator(positiveData,[],@mean,3e-1);
boundaryData=testData(boundaryLabel==1,:);
boundaryData=consolidator(boundaryData,[],@mean,1e-1);
save boundary_data3 boundaryData;

% Boundary illustration 
figure(1);clf;
plot(positiveData(:,1),positiveData(:,2),'r*');hold on;
plot(negativeData(:,1),negativeData(:,2),'b*');hold on;
plot(testData(boundaryLabel==1,1),testData(boundaryLabel==1,2),'g*','linewidth',2);

%% Time-domain validation
%
clear ibrlData;
load ibrl_data;
index=date*24*60*60+time;
ibrlData=[month index moteid temperature humidity];
ibrlData(ibrlData(:,1)~=3,:)=[];  

% 
clear predictLabel moteData;
for i=37
    moteData{i}=ibrlData(ibrlData(:,3)==i,[2 4 5]);
    predictLabel{i}=svdd_classify(ocSVM,moteData{i}(:,[2 3]));
    figure(i);clf;
    plot(moteData{i}(:,[2 3]),'b-');
    hold on;
    plot(find(predictLabel{i}==-1),moteData{i}(predictLabel{i}==-1,[2 3]),...
        'ro','linewidth',2);
end

moteData=moteData{i}(:,[2 3]);
anomalyIndex=find(predictLabel{i}==-1);
anomalyData=moteData(anomalyIndex,:);
save timeData37 moteData anomalyIndex anomalyData;