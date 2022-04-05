
import torchvision
import torch
import os
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224))])
def get_torch_creature_data(filename="Flickr8k.token.modified.txt"):
    '''
        This function will return the original imgs from the creature folder in the list of tensor
        and the caption they correspond in a list of strings
    '''
    creature_path = "./data/creatures/"
    creature_f = open(creature_path + filename)
    lines = creature_f.readlines()
    data = [line.split(maxsplit = 1) for line in lines]
    picture_ids = []
    captions = []
    for i in range(0,len(data),5):
        picture_ids.append(data[i][0][:-2])
        captions.append(data[i][1][:-2])
    
    creature_imgs = [transform(torchvision.io.read_image(creature_path +'Flicker8k_Dataset/' +i ,mode = torchvision.io.ImageReadMode.RGB)).float().unsqueeze(0) for i in picture_ids]
    torch_creature_imgs = torch.cat(creature_imgs)
    return (torch_creature_imgs, captions)
    
def get_torch_item_data():
    '''
        This function will return the resized(224x224) imgs from the item folder in tensor
        and the caption they correspond in a list of strings
    '''
    
    item_path = "./data/items/"
    item_f = open(item_path + "etsy.txt",encoding="latin-1")
    lines = item_f.readlines()
    data = [line.split(maxsplit = 1) for line in lines]
    picture_ids = []
    captions = []
    for i in range(len(data)):
        picture_ids.append(data[i][0])
        captions.append(data[i][1][:-2])
    
    item_imgs = [transform(torchvision.io.read_image(item_path + 'images/images/' + i,mode = torchvision.io.ImageReadMode.RGB)).float().unsqueeze(0) for i in picture_ids]
    torch_item_imgs = torch.cat(item_imgs)
    return (torch_item_imgs, captions)


def get_creature_data(filename="Flickr8k.token.modified.txt"):
    '''
        This function will return the resized(224x224) imgs from the creature folder in tensor
        and the caption they correspond in a list of strings
    '''
    creature_path = "./data/creatures/"
    creature_f = open(creature_path + filename)
    lines = creature_f.readlines()
    data = [line.split(maxsplit = 1) for line in lines]
    picture_ids = []
    captions = []
    for i in range(0,len(data),5):
        picture_ids.append(data[i][0][:-2])
        captions.append(data[i][1][:-2])
    
    creature_imgs = [torchvision.io.read_image(creature_path +'Flicker8k_Dataset/' +i ,mode = torchvision.io.ImageReadMode.RGB).float() for i in picture_ids]

    return (creature_imgs, captions)
    
def get_item_data():
    '''
        This function will return the imgs from the item folder in tensor
        and the caption they correspond in a list of strings
    '''
    
    item_path = "./data/items/"
    item_f = open(item_path + "etsy.txt",encoding="latin-1")
    lines = item_f.readlines()
    data = [line.split(maxsplit = 1) for line in lines]
    picture_ids = []
    captions = []
    for i in range(len(data)):
        picture_ids.append(data[i][0])
        captions.append(data[i][1][:-2])
    
    item_imgs = [torchvision.io.read_image(item_path + 'images/images/' + i,mode = torchvision.io.ImageReadMode.RGB).float() for i in picture_ids]

    return (item_imgs, captions)


if "__init__" == "__main__":
    print("loading")
    item_img, item_caption = get_item_data()
    creature_img, creature_caption = get_creature_data()
    print("done")