# Model description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

# Resources
    A blog post on how to [Fine-Tune ViT for Image Classification with Hugging Face Transformers](#https://huggingface.co/blog/fine-tune-vit)
    A blog post on Image Classification with Hugging Face Transformers and Keras
    A notebook on Fine-tuning for Image Classification with Hugging Face Transformers
    A notebook on how to Fine-tune the Vision Transformer on CIFAR-10 with the Hugging Face Trainer
    A notebook on how to Fine-tune the Vision Transformer on CIFAR-10 with PyTorch Lightning

⚗️ Optimization

    A blog post on how to Accelerate Vision Transformer (ViT) with Quantization using Optimum

⚡️ Inference

    A notebook on Quick demo: Vision Transformer (ViT) by Google Brain

🚀 Deploy

    A blog post on Deploying Tensorflow Vision Models in Hugging Face with TF Serving
    A blog post on Deploying Hugging Face ViT on Vertex AI
    A blog post on Deploying Hugging Face ViT on Kubernetes with TF Serving
