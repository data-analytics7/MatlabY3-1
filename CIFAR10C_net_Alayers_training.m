function [] = CIFAR10C_net_Alayers()

clc
clear all

rng(10)
warning('on','nnet_cnn:warning:GPULowOnMemory')


global traindatatype
global epoch_groups

global opts
global basicepochs
global batchsize
global maxloop


path = 'C:\Image Classification Project\ConvNet\Test images';
% global imall
% global blurimall

% cifar10Data = tempdir;
%
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
%
% helperCIFAR10Data.download(url,cifar10Data);
% [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);

%%%%%%%%%%%%%% Loading CIFAR 10 data %%%%%%%%%%%%%%%%%%%%
% cifar10Data = 'C:\datasets\cifar-10-batches-mat\';
% %
% load(sprintf('%sdata_batch_1.mat',cifar10Data))
% traini = data;
% trainingLabels = labels;
% load(sprintf('%sdata_batch_2.mat',cifar10Data))
% traini = [traini; data];
% trainingLabels = [trainingLabels; labels];
% load(sprintf('%sdata_batch_3.mat',cifar10Data))
% traini = [traini; data];
% trainingLabels = [trainingLabels; labels];
% load(sprintf('%sdata_batch_4.mat',cifar10Data))
% traini = [traini; data];
% trainingLabels = [trainingLabels; labels];
% load(sprintf('%sdata_batch_5.mat',cifar10Data))
% traini = [traini; data];
% trainingLabels = [trainingLabels; labels];
% load(sprintf('%stest_batch.mat',cifar10Data))
% testi = data;

% cat_trainingLabels = categorical(trainingLabels);
% cat_testLabels = categorical(labels);

% trainingImages = reshape(traini',[32 32 3 50000]);
% testImages = reshape(testi',[32 32 3 10000]);

% trainingImages_orig = permute(trainingImages,[2 1 3 4]);
% testImages_orig = permute(testImages, [2 1 3 4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

batchsize = 256;

imageSize = [32  32 3];


ALayers = [
    imageInputLayer(imageSize,'Name','input');
    dropoutLayer(0.2,'Name','dropout0');
    
    convolution2dLayer([5 5], 96, 'Padding', 2, 'Stride',2,'Name','conv1')
    reluLayer('Name','relu1')
    dropoutLayer(0.5,'Name','dropout1');
       
    convolution2dLayer([5 5], 192, 'Padding', 2,'Stride',2, 'Name','conv2')
    reluLayer('Name','relu2')
    dropoutLayer(0.5,'Name','dropout2');
       
    convolution2dLayer([3 3], 192, 'Padding', 0, 'Stride',1,'Name','conv3')
    reluLayer('Name','relu3')
    
    convolution2dLayer([1 1], 192,'Padding', 0, 'Stride',1,'Name','conv4')
    reluLayer('Name','relu4')
    
    convolution2dLayer([1 1], 10,'Padding', 0, 'Stride',1,'Name','conv5')
    reluLayer('Name','relu5')
    
    averagePooling2dLayer(6,'Name','avgpool1')
   
    softmaxLayer('Name','softmax1')
    classificationLayer('Name','classify1')
    
    ];


basicepochs = 5;
maxloop = 80;
setopts(1)

%%%%%%%%% TRAIN Cifar 10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% for loop=1:maxloop
% 
%     epoch_cnt = loop*basicepochs;
%     if (mod(epoch_cnt,50)==0)
%         setopts(epoch_cnt);
%     end
%     
%     if  (loop==1)
%         cifar10Net = trainNetwork(trainingImages_orig, cat_trainingLabels, ALayers, opts);
%     else
%         cifar10Net = trainNetwork(trainingImages_orig, cat_trainingLabels, cifar10Net.Layers, opts);
%     end
%      fname = sprintf('TrainedNet_%d.mat',loop*basicepochs);
%     save(fname,'cifar10Net');
%     %%%%
% 
%     fname = sprintf('TrainedNet_%d.mat',loop*basicepochs);
%     load(fname,'cifar10Net');
% 
%     YPred = classify(cifar10Net,trainingImages_orig);
%     accuracy_train = (sum(cat_trainingLabels==YPred)/numel(cat_trainingLabels));
%     
%     YPred = classify(cifar10Net,testImages_orig);
%     accuracy_test = (sum(cat_testLabels==YPred)/numel(cat_testLabels));
% 
%     fname = sprintf('TestAccuracy_%d.mat',loop*basicepochs);
%     save(fname,'accuracy_train','accuracy_test');
% 
%     [epoch_cnt, accuracy_train, accuracy_test]
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% %%%%% CIFAR 10 TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     fname = sprintf('TrainedNet_20.mat');
%     load(fname,'cifar10Net');
%     YPred = classify(cifar10Net,testImages)
% %     accuracy = (sum(testLabels==YPred)/numel(testLabels))
%
%     classnames = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
%
%     [ff, classnames(1,int8(YPred)+1)]
%
%
%     figure(1)
%     montage(testImages(:,:,:,1:10))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%Own Images TEST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fname = sprintf('TrainedNet_350.mat');
load(fname,'cifar10Net');

fnames = dir(sprintf('%s/*.jpg',path))


result = cell(length(fnames),2)
imagearr = zeros(32,32,3,length(fnames));

classnames = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};


for loop=1:length(fnames)
    onefilename =fnames(loop,1).name;
    
    
    onefilepath = sprintf('%s/%s',path,onefilename);
    ownimage = imread(onefilepath);
    
    resown = imresize((ownimage),[32 32]);
%     
%     
%     
%     figure(1)
%     subplot(121)
%     imshow(ownimage);
%     subplot(122)
%     imshow(resown)
%     
    tic()
    YPred = classify(cifar10Net,resown)
    toc()
%     %     accuracy = (sum(testLabels==YPred)/numel(testLabels))
%     
%     
    result(loop,:) = [onefilename, classnames(1,int8(YPred))];
    imagearr(:,:,:,loop) = resown;
% end
% 
save('Allresults.mat','result');
% 
% figure(100)
% montage(uint8(imagearr),'Size',[5,8]);


% %%%%% PLOTTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% plot_graphs()



end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [] = setopts(epoch_cnt)

global opts
global basicepochs
global batchsize


if(epoch_cnt < 200)
    
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.01, ...%0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 200, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', basicepochs , ... %% NOTE  this
    'MiniBatchSize', batchsize, ...
    'Verbose', true,...
   'ExecutionEnvironment','multi-gpu');%,...
 % 'Plots','training-progress');
% 'OutputFcn',@(info)savetrainingvals(info));

elseif ((epoch_cnt >= 200) & (epoch_cnt < 250))
    
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...%0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 200, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', basicepochs , ... %% NOTE  this
    'MiniBatchSize', batchsize, ...
    'Verbose', true,...
   'ExecutionEnvironment','multi-gpu');%,...
 % 'Plots','training-progress');
% 'OutputFcn',@(info)savetrainingvals(info));

elseif  ((epoch_cnt >= 250) & (epoch_cnt < 300))
    
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.0001, ...%0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 200, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', basicepochs , ... %% NOTE  this
    'MiniBatchSize', batchsize, ...
    'Verbose', true,...
   'ExecutionEnvironment','multi-gpu');%,...
 % 'Plots','training-progress');
% 'OutputFcn',@(info)savetrainingvals(info));

elseif  ((epoch_cnt >= 300) & (epoch_cnt <= 400))
    
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.00001, ...%0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 200, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', basicepochs , ... %% NOTE  this
    'MiniBatchSize', batchsize, ...
    'Verbose', true,...
    'ExecutionEnvironment','multi-gpu');%,...
 % 'Plots','training-progress');
% 'OutputFcn',@(info)savetrainingvals(info));
end


%end


% function plot_graphs()

% global maxloop
% global basicepochs

% maxloop = 70

% res =zeros(2,maxloop)

% for loop=1:maxloop
    % fname = sprintf('TestAccuracy_%d.mat',loop*basicepochs);
    % load(fname,'accuracy_train','accuracy_test');

    % epoch_cnt = loop*basicepochs;
    % [epoch_cnt, accuracy_train, accuracy_test]
    
    % res(:,loop) = [accuracy_train, accuracy_test];
% end
    
% figure(1)
% plot([1:maxloop].*basicepochs, res','.-')
% legend('Training', 'Testing')
% xlabel('Epochs')
% ylabel('Accuracy')
% axis([1,maxloop.*basicepochs,0,1])
% end

