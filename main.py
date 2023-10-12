from datasets import load_dataset
from input_processor import InputProcessor, plot_images
import numpy as np

if __name__ == '__main__':
    model_path = "facebook/data2vec-vision-base"
    dataset = load_dataset("ceyda/smithsonian_butterflies")
    all_images = dataset['train'][0:500]['image']

    random_image = all_images[np.random.randint(len(all_images))]
    print(random_image)

    # keep the original image in the list of similar images
    # so that we know it works correctly
    input_processor = InputProcessor(model_path, all_images)

    similar_images = input_processor.process_input(random_image, threshold=0.99995)
    print("number of similar images:", len(similar_images))
    plot_images(random_image, similar_images)
