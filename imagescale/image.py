from PIL import Image

# List of image filenames
image_files = ["car.jpg", "deer.jpg", "plane.jpg", "horse.jpg"]

# Define the new size
new_size = (32, 32)

# Resize and save the images
for image_file in image_files:
    img = Image.open(image_file)
    img_resized = img.resize(new_size, Image.LANCZOS)
    img_resized.save(f"resized_{image_file}")

print("Images have been resized successfully.")
