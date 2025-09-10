pip install diffusers transformers accelerate torch safetensors
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Get input prompt from the user
user_prompt = input("Enter a prompt for image generation: ")

# Generate the image based on the user's prompt
try:
    with torch.autocast("cuda"):
        image = pipe(user_prompt).images[0]

    # Display the generated image
    plt.imshow(image)
    plt.axis("off")  # Remove axis labels
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
