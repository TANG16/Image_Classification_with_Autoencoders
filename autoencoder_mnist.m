%Autoencoder using MNIST dataset. Loader functions provided by Stanford.
train_images = loadMNISTImages('train-images.idx3-ubyte'); % get train images  
train_labels = loadMNISTLabels('train-labels.idx1-ubyte'); % get train labels

%the digit 0 will be represented by 10th class 
train_labels(train_labels==0) = 10;                   
%Change labels to one hot vector
train_labels = dummyvar(train_labels);                          
train_labels = train_labels';

test_images = loadMNISTImages('t10k-images.idx3-ubyte'); % get test images  
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % get test labels

test_labels(test_labels==0) = 10;                               
test_labels=dummyvar(test_labels); 
test_labels = test_labels';  


imageWidth = 28;
imageHeight = 28;

%VIsualizing the first image
first_image = train_images(:,1);
first_image = reshape(first_image,imageWidth,imageHeight);
%Magnifying image
first_image = imresize(first_image,[100,100]);
figure()
imshow(first_image);

rng('default');
hiddenSize1 = 150;
autoenc1 = trainAutoencoder(train_images,hiddenSize1, ...
    'MaxEpochs',10, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)
figure()
plotWeights(autoenc1);
feat1 = encode(autoenc1,train_images);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',20, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);
softnet = trainSoftmaxLayer(feat2,train_labels,'MaxEpochs',400);
deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)
y = deepnet(test_images);
plotconfusion(test_labels,y);

deepnet = train(deepnet,train_images,train_labels);
y = deepnet(test_images);
plotconfusion(test_labels,y);