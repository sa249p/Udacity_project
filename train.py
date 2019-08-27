def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision import datasets, transforms, models
    import torchvision.models as models
    from PIL import Image
    import json
    from matplotlib.ticker import FormatStrFormatter
    import collections
    import seaborn as sb

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing    sets
    #data_transforms = 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    #image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir,            transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(len(cat_to_name))
    print(cat_to_name)

    # TODO: Build and train your network
    model_vgg = models.vgg16(pretrained=True)
    model_vgg

    #Freezing parameters
    for parameters in model_vgg.parameters():
        parameters.requires_grad = False
    
    classifier = torch.nn.Sequential(collections.OrderedDict([
                          ('fc1', torch.nn.Linear(25088, 4096, bias=True)),
                          ('relu1', torch.nn.ReLU()),
                          ('dropout1', torch.nn.Dropout(p=0.2)),
                          ('fc2', torch.nn.Linear(4096, 102, bias=True)),
                          ('output', torch.nn.LogSoftmax(dim=1))
                          ]))
    
    model_vgg.classifier = classifier

    model_vgg

    #automatically use cuda if enabled
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device

    #change model now 
    model_vgg.to(device)

    #initializing loss and optimizer 
    cri = torch.nn.NLLLoss()
    opt = torch.optim.Adam(model_vgg.classifier.parameters(),lr=0.001)

    epochs = 5
    img_no = 25
    steps = 0

    def validation(model_vgg, testloader, cri):
        testloss = 0
        accuracy = 0
        for count, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model_vgg.forward(inputs)
            testloss += cri(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        return testloss, accuracy

    print("Train started")

    for subs in range(epochs):
        runningloss = 0
        model_vgg.train()
        for count, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model_vgg.forward(inputs)
            loss = cri(outputs, labels)
            loss.backward()
            opt.step()
        
            runningloss += loss.item()
        
        if steps % img_no == 0:
            model_vgg.eval()

            with torch.no_grad():
                validloss, accuracy = validation(model_vgg, validloader, cri)
            
            print(subs+1, epochs)
            print('Training Loss: ',(runningloss/img_no))
            print('Validation Loss: ',(validloss/len(testloader)))
            print('Accuracy: ', accuracy/len(testloader))
            runningloss = 0
            model_vgg.train()

    print("Train Done!")

    # TODO: Do validation on the test set
    right = 0
    total = 0

    with torch.no_grad ():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model_vgg (inputs)
            _, predicted = torch.max (outputs.data,1)
            total += labels.size (0)
            right += (predicted == labels).sum().item()

        print('Accuracy = ',(right*100/total),'%')

    # TODO: Save the checkpoint 
    model_vgg.class_to_idx = train_data.class_to_idx

    checkpoint = {'classifier': model_vgg.classifier,
              'state_dict': model_vgg.state_dict (),
              'mapping':    model_vgg.class_to_idx
             }        

    torch.save (checkpoint, 'project_checkpoint.pth')
if __name__ == '__main__': main()

