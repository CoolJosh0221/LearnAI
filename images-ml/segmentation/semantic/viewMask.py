import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from getFiles import target_paths, input_img_paths


def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    plt.show()


if __name__ == "__main__":
    plt.axis("off")
    plt.imshow(load_img(input_img_paths[9]))
    plt.show()

    print("test 1")
    img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
    display_target(img)
