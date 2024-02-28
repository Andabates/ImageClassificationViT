from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image as img
from IPython.display import Image, display
FILE_NAME = 'image1.jpeg'
display(Image(FILE_NAME, width = 700, height = 400))
image_array = img.open('image1.jpeg')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images = image_array, 
                           return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print(predicted_class_idx)
print("Predicted class:", model.config.id2label[predicted_class_idx])