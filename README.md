# Model description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

# Resources
A blog post on how to [Fine-Tune ViT for Image Classification with Hugging Face Transformers](https://huggingface.co/blog/fine-tune-vit)/n
A blog post on [Image Classification with Hugging Face Transformers and Keras](https://www.philschmid.de/image-classification-huggingface-transformers-keras)
A notebook on [Fine-tuning for Image Classification with Hugging Face Transformers](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with the Hugging Face Trainer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb)
A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with PyTorch Lightning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)

Optimization
A blog post on how to [Accelerate Vision Transformer (ViT) with Quantization using Optimum](https://www.philschmid.de/optimizing-vision-transformer)

Inference
A notebook on Quick demo: [Vision Transformer (ViT) by Google Brain](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb)

Deploy
A blog post on [Deploying Tensorflow Vision Models in Hugging Face with TF Serving](https://huggingface.co/blog/tf-serving-vision)
A blog post on [Deploying Hugging Face ViT on Vertex AI](https://huggingface.co/blog/deploy-vertex-ai)
A blog post on [Deploying Hugging Face ViT on Kubernetes with TF Serving](https://huggingface.co/blog/deploy-tfserving-kubernetes)
