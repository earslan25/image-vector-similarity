from transformers import AutoFeatureExtractor, AutoModel, AutoImageProcessor, ViTMAEForPreTraining
import torch
import torchvision.transforms as T
import numpy as np


class VectorEmbedder:
    def __init__(self, model_path, device="mps"):
        # Initialize the vector embedding model
        # Load the model from the specified path
        device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        # self.model = ViTMAEForPreTraining.from_pretrained(model_path).to(device)
        # self.processor = AutoImageProcessor.from_pretrained(model_path)
        extractor = AutoFeatureExtractor.from_pretrained(model_path)
        # print(extractor)
        # if "shortest_edge" in extractor.size:
        #     dimension = (extractor.size["shortest_edge"], extractor.size["shortest_edge"])
        # else:
        #     dimension = (extractor.size["height"], extractor.size["width"])
        self.transform = T.Compose([
            # T.Resize(extractor.size["shortest_edge"]),
            T.Resize((extractor.size["height"], extractor.size["width"])),
            # T.CenterCrop(extractor.size["shortest_edge"]),
            T.CenterCrop(extractor.size["height"]),
            T.ToTensor(),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ])

    def embed_data(self, images):
        # Embed the images into a vector
        # images: a list of images
        # return: a list of vectors
        # Transform the images to tensors
        images_transformed = torch.stack(
            [self.transform(image) for image in images]
            # [self.processor(image, return_tensors="pt").pixel_values for image in images]
        )
        print("done transforming images")
        print(images_transformed.shape)

        # Embed the images
        new_images = {"pixel_values": images_transformed.to(self.model.device)}
        with torch.no_grad():
            embeddings = self.model(**new_images).last_hidden_state[:, 0].cpu()
            # embeddings = self.model(**new_images).logits.cpu()
        return torch.from_numpy(np.array(embeddings))
