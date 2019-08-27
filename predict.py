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
# TODO: Write a function that loads a checkpoint and rebuilds the model
    def checkpointload(filepath):
        checkpoint = torch.load(filepath)
        model_vgg = models.vgg16(pretrained=True);
        for param in model_vgg.parameters():param.requires_grad = False
        model_vgg.classifier = checkpoint['classifier']
        model_vgg.load_state_dict(checkpoint['state_dict'])
        model_vgg.class_to_idx = checkpoint['mapping']        
        return model_vgg
    #model_vgg = checkpointload()
    model_vgg = checkpointload ('project_checkpoint.pth')
    #print(model_vgg)
    
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
    # TODO: Process a PIL image for use in a PyTorch model
        img = Image.open(image)
        adjustments = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
        img_transpose = adjustments(img)
    
        return img_transpose


    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
    
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
    
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
    
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
    
        ax.imshow(image)
    
        return ax


    #imshow(process_image("flowers/test/15/image_06351.jpg"))

    def predict(image_path, model_vgg, top_k=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
    
    # TODO: Implement the code to predict the class from an image file
        model_vgg.to("cpu")
        model_vgg.eval()
        img = process_image(image_path)
    #Converting
        img_torch =     torch.from_numpy(np.expand_dims(img,axis=0)).type(torch.FloatTensor).to("cpu")
        prob = model_vgg.forward(img_torch) #log scale
        #Linear scale
        prob_linear = torch.exp(prob)
    
        top_match, top_labels = prob_linear.topk(top_k)
    
        top_match = np.array(top_match.detach())[0] 
        top_labels = np.array(top_labels.detach())[0]

        idx_to_class = {val: key for key, val in    
                                      model_vgg.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labels]
        top_flowers = [cat_to_name[lab] for lab in top_labels]
    
        return top_match, top_labels, top_flowers

# TODO: Display an image along with the top 5 classes
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(len(cat_to_name))
    print(cat_to_name)
    image_path = "flowers/test/15/image_06351.jpg"
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    img = process_image(image_path)
    #imshow(img, ax, title = title_);
    probs, labs, flowers = predict(image_path, model_vgg) 
    print(probs,flowers)
    #plt.subplot(2,1,2)
    #sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
    #plt.show()

    # TODO: Display an image along with the top 5 classes

    image_path = "flowers/test/3/image_06634.jpg"
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    img = process_image(image_path)
    #imshow(img, ax, title = title_);
    probs, labs, flowers = predict(image_path, model_vgg) 
    print(probs,flowers)
    #plt.subplot(2,1,2)
    #sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
    #plt.show()

    image_path = "flowers/test/47/image_04966.jpg"
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    img = process_image(image_path)
    #imshow(img, ax, title = title_);
    probs, labs, flowers = predict(image_path, model_vgg) 
    print(probs,flowers)
    #plt.subplot(2,1,2)
    #sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
    #plt.show()


if __name__ =='__main__': main()