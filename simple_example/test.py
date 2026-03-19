# --- Optional visualization (matplotlib) ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow, show
except ImportError:
    raise ImportError(
        "Matplotlib is required for visualization.\n"
        "Install it with:\n"
        '    pip install "winsorcam[simple_vis] @ git+https://github.com/USD-AI-ResearchLab/Winsor-CAM.git"'
    )

# --- Your own package ---
try:
    from winsorcam import WinsorcamClass
except ImportError:
    raise ImportError(
        "The 'winsorcam' package is not installed.\n"
        "Install it with:\n"
        '    pip install "winsorcam @ git+https://github.com/USD-AI-ResearchLab/Winsor-CAM.git"'
    )

from pathlib import Path

# --- Required: PyTorch + torchvision ---
try:
    import torch
    from torchvision.models import densenet121
    from torchvision import transforms
    import torchvision.io as tvio
except ImportError:
    raise ImportError(
        "PyTorch and torchvision are required but not installed.\n\n"
        "Install the correct version for your system from:\n"
        "    https://pytorch.org/get-started/locally/\n\n"
        "Or install via pip with the appropriate CUDA/CPU flag, for example:\n"
        "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
        "(Replace cu121 with the version matching your hardware.)"
    )

# load densenet model with imageNet weights
model = densenet121(weights="IMAGENET1K_V1")

# get all convolutional layer names in the model
model_usable_layer_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) and "AuxLogits" not in name]

# create an instance of the WinsorCAM class
winsorcam = WinsorcamClass(model, model_usable_layer_names)
del model  # delete the original model to free up memory since we have it in the winsorcam class now
# state what layers we want to use for the WinsorCAM explanation but with "model." in front of them to match the layer names in the model
# so lets modify model_usable_layer_names
model_usable_layer_names = [f"model.{name}" for name in model_usable_layer_names]

# load an image and preprocess it to be the right size for the model
image_path = Path("./test_image.png")
image = tvio.read_image(str(image_path)).float()

# now we need to transform using a 224x224 resize and the standard imagenet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x / 255.0),  # Scale to [0, 1] before normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# apply the transform to the image and add a batch dimension
input_tensor = transform(image).unsqueeze(0)  # add batch dimension

winsorcam.eval()  # set model to evaluation mode
winsorcam.model.eval()  # set the internal model to evaluation mode as well

# get the predicted class using the model
with torch.no_grad():
    output = winsorcam.model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted class: {predicted_class}")

# In this case want to use all the layers available in the model for the explanation
# If other layers are desired then you will need to specify them here by indexes
desired_layer_names = model_usable_layer_names[:]


# now we that we have the predicted class, we can generate the gradcams and importance scores for that class
stacked_gradcam, gradcams, importance_tensor = winsorcam.get_gradcams_and_importance(
    input_tensor=input_tensor,          # the input tensor to explain
    target_class=predicted_class,       # the class index to explain
    layers=desired_layer_names,         # the layers to use for the explanation
    gradient_aggregation_method='mean', # method for aggregating gradients within the layers (mean is typically the only used option)
    layer_aggregation_method='mean',    # method for aggregating importance scores across layers (mean or max are common options)
    stack_relu=True,                    # whether to apply ReLU to the layer importance scores (should typically be True otherwise large negative importance per layer importance can modify)
                                        # implimetation of layer_relu=False has not been fully tested
    interpolation_mode='bilinear'       # the interpolation mode to use when resizing the gradcams to the input image size
                                        # this heavily impacts the spatial aspects of the explanation
                                        # only pytorch interpolation modes are available
    )

# now we create the winsorized version of the stacked gradcam
winsor_gradcam, winsor_importance = winsorcam.winsorize_stacked_gradcam(
        input_tensor, stacked_gradcam, importance_tensor,
        interpolation_mode="bilinear",
        winsor_percentile=90 # the percentile to use for winsorization
                             # use values between 0 and 100
                             # where higher values will result in layers with higher
                             # importance having more of their values preserved
                             # while lower value result in layers being weighted more equally
    )


# lets make a simple plot of 2x5 with the original image, the the winsorcam explanation
# normal final layer gradcam, the image overlayed with the winsorcam explanation, and the image overlaid with the normal gradcam
# underneath of the winsorcams I want a plot of the 

#first resize the original image to be the same size as the winsorcam explanation
image = transforms.Resize(winsor_gradcam.shape[1:])(image)

fig, axs = plt.subplots(1, 5, figsize=(20, 5))
# original image
axs[0].imshow(image.permute(1, 2, 0).cpu().numpy().astype(int))
axs[0].set_title("Original Image")

# winsorcam explanation
axs[1].imshow(winsor_gradcam.squeeze().cpu().numpy(), cmap='nipy_spectral')
axs[1].set_title("WinsorCAM Explanation")

# normal final layer gradcam
axs[2].imshow(stacked_gradcam[-1].squeeze().cpu().numpy(), cmap='nipy_spectral')
axs[2].set_title("Final Layer GradCAM")

# image overlayed with winsorcam explanation
axs[3].imshow(image.permute(1, 2, 0).cpu().numpy().astype(int))
axs[3].imshow(winsor_gradcam.squeeze().cpu().numpy(), alpha=0.5, cmap='nipy_spectral')
axs[3].set_title("Image with WinsorCAM Overlay")

# image overlayed with normal gradcam
axs[4].imshow(image.permute(1, 2, 0).cpu().numpy().astype(int))
axs[4].imshow(stacked_gradcam[-1].squeeze().cpu().numpy(), alpha=0.5, cmap='nipy_spectral')
axs[4].set_title("Image with Final Layer GradCAM Overlay")

# set a tight layout so the titles are not overlapping with the images
plt.tight_layout()

# save the figure in the current directory
plt.savefig("winsorcam_example.png")
plt.show()
