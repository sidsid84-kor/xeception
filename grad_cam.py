import keras
import matplotlib.cm as cm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    import cv2
    from tensorflow.keras.utils import plot_model


def grad_cam(img, img_scaled, model, last_conv_layer_name, show_grad_cam=True, alpha=0.4, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_scaled)
        if len(preds) > 1: ##### 보조분류기로 인해 아웃풋이 여러개일 경우 0번째 index의 output을 가져옴
            preds = preds[0]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    superimposed_img = __superimposed_img(img, heatmap, alpha=alpha)
    if show_grad_cam:
        plt.figure(figsize=(10, 10))
        plt.imshow(superimposed_img)
        plt.show()
    return superimposed_img

def __superimposed_img(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

if __name__ == "main":
    model = load_model(r'C:\Users\CGAC\Desktop\Server\predict_models\classes\DA4076_defect.h5')
    ### 마지막 cnn layer층 이름 확인용 ###
    a = model.layers
    for layer in a:
        print(layer.name)
    plot_model(model, show_layer_names=True)
    ###################################

    img = cv2.imread(r'D:\kgn\_forPaper\train\6_line\0_20220122051319_4-0.bmp')
    def preprocessing(image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_rgb, dsize=(600, 600))
            image_scaled = image_resize / 255.
            image_reshape = image_scaled.reshape(1, 600, 600, 3)
            dst = np.asarray(image_reshape, dtype=np.float32)
            return dst
        except Exception as e:
            print(e.__str__())
            print('image preprocessing failed.')
    img_scaled = preprocessing(img)

    grad_cam = grad_cam(img=img, img_scaled=img_scaled, show_grad_cam = True, model=model, last_conv_layer_name='concatenate_26')