import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def process_image(image_path):
   
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = preprocess(image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image

def predict(image_path, model, device, topk=5, cat_to_name=None):
    
    img_tensor = process_image(image_path)

    
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
    probabilities, indices = torch.topk(torch.softmax(output, dim=1), topk)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    probabilities = probabilities.squeeze().cpu().numpy().tolist()

    if cat_to_name:
        class_names = [cat_to_name[class_] for class_ in classes]
    else:
        class_names = classes

    return probabilities, class_names

def load_model(file_path):
    checkpoint = torch.load(file_path)
    model = models.alexnet(pretrained=True)
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained neural network.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("model_checkpoint", help="Path to the model checkpoint file.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show.")
    parser.add_argument("--category_names", help="Path to a JSON file mapping category indices to category names.")

    args = parser.parse_args()
    loaded_model = load_model(args.model_checkpoint)
    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    probabilities, classes = predict(args.image_path, loaded_model, device='cuda:0', topk=args.topk, cat_to_name=cat_to_name)
    print("Probabilities:", probabilities)
    print("Predicted Classes:", classes)
