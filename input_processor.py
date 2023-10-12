from query_comparator import QueryComparator
from vector_embedder import VectorEmbedder
import matplotlib.pyplot as plt
import numpy as np


class InputProcessor:
    def __init__(self, model_path, queries):
        # model_path: the path to the model to use for vector embedding
        # queries: the queries to compare the input to
        self.embedder = VectorEmbedder(model_path)
        print("done initializing embedder")
        self.original_queries = queries
        transformed_queries = self.embedder.embed_data(self.original_queries)
        print(transformed_queries.shape)
        print("done embedding queries")
        self.comparator = QueryComparator(transformed_queries)

    def process_input(self, image, threshold=0.5):
        # image: the image to process
        # return: the images that are similar to the image
        transformed_image = self.embedder.embed_data([image])
        print(transformed_image.shape)
        similar_indices = self.comparator.compare(transformed_image, threshold)
        return [self.original_queries[i] for i in similar_indices]


def plot_images(original_image, similar_images):
    # images: a list of images
    # Plot the images
    if len(similar_images) == 0:
        print("No similar images found")
        return
    images = [original_image] + similar_images
    fig, axes = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis("off")
        if i == 0:
            axes[i].set_title("Og")
        else:
            axes[i].set_title(str(i))
    plt.show()

