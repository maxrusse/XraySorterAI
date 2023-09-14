import tensorflow as tf
import cv2
import numpy as np
from tf_explain.core.grad_cam import GradCAM
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import warnings
import nibabel as nib
import numpy as np
warnings.simplefilter(action='ignore', category=Warning)

def load_and_align_nifti(file_path):
    image = nib.load(file_path)
    aligned_image = align_image_coordinates(image)
    return aligned_image

def align_image_coordinates(image):
    affine_matrix = image.affine
    max_indices = np.argmax(np.abs(affine_matrix[0:3, 0:3]), axis=0)
    signs = np.sign(affine_matrix[max_indices, np.arange(3)])

    permutation_matrix = np.eye(4)
    for idx in range(3):
        permutation_matrix[idx, max_indices[idx]] = 1

    data = image.get_fdata()
    inverse_indices = list(np.argsort(max_indices))
    if len(image.shape) > 3:
        inverse_indices += list(range(3, len(image.shape)))

    transposed_data = np.transpose(data, inverse_indices)

    flip_matrix = np.eye(4)
    for idx in range(3):
        flip_matrix[idx, idx] = signs[inverse_indices[idx]]
        if signs[inverse_indices[idx]] < 0:
            flip_matrix[idx, 3] = transposed_data.shape[idx] - 1
            transposed_data = np.flip(transposed_data, axis=idx)

    new_affine = np.matmul(np.matmul(affine_matrix, permutation_matrix), flip_matrix)
    image.set_sform(new_affine)

    return nib.nifti1.Nifti1Image(transposed_data, new_affine, header=image.header)


TAGS_PROJ = ['ap_clavicle','obl_clavicle','ap_ac','ap_shoulder','y-view','axial_shoulder','ap_elbow','lat_elbow','radial_head',
             'ap_wrist','lat_wrist','ap_hand','obl_hand','ap_finger','lat_finger','ap_thumb','lat_thumb','ap_leg','ap_pelvis',
             'ap_hip','Lauenstein','ap_knee','lat_knee','defile','ap_ankle','lat_ankle','lat_calcaneus','axial_calcaneus',
             'ap_foot','lat_foot','obl_foot','ap_forefoot','obl_forefoot','ap_toe','lat_toe','ap_btoe','lat_btoe','ap_cspine',
             'lat_cspine','dens','ap_tspine','lat_tspine','ap_lspine','lat_lspine','nasal_bone']

TAGS_SIDE = ['bodyside_right', 'bodyside_left'] * 6

def preprocess_image(image_tensor, dimensions=(256, 256)):
    image_array = tf.squeeze(image_tensor).numpy()
    normalized_image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    blurred_image = cv2.medianBlur(normalized_image, 3)
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    inverse_image = 255 - image_array
    rgb_image = np.dstack((thresholded_image, image_array, inverse_image))
    resized_image = cv2.resize(rgb_image, dimensions, interpolation=cv2.INTER_CUBIC)
    return cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def predict_image(image_tensor, model_path):
    with tf.device("/cpu:0"):
        model = tf.keras.models.load_model(model_path)
 
        predictions = model.predict(image_tensor,verbose=0,)
        explainer = GradCAM()
        visualization = explainer.explain(
            [image_tensor.numpy(), predictions.argmax()], model, 
            class_index=predictions.argmax(), colormap=cv2.COLORMAP_JET, 
            image_weight=1
        )
    return predictions, visualization

def predict_from_file_path(file_path, model):
    with tf.device("/cpu:0"):
        nifti_image = load_and_align_nifti(file_path)
        image_data = nifti_image.get_fdata()
        if image_data.ndim == 4:
            image_data = image_data[:, :, :, 1]
        
        original_image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
        preprocessed_image = preprocess_image(original_image_tensor)
        preprocessed_tensor = tf.expand_dims(preprocessed_image, axis=0)
        preprocessed_tensor = tf.cast(preprocessed_tensor, tf.float32)
        preprocessed_tensor = tf.keras.applications.xception.preprocess_input(preprocessed_tensor)
        
        return predict_image(preprocessed_tensor, model)

def visualize_prediction_projection_colab(file_path):
    """
    Visualize predictions for a NIFTI image file using the projection model.

    Args:
        file_path (str): Path to the NIFTI image file.
    """
    print('----------------------------------------- Projection -----------------------------------------')

    predictions, visualization = predict_from_file_path(file_path, 'XraySorterAI/models/model_proj')
    prediction_tag = TAGS_PROJ[predictions.argmax()]

    plt.figure(figsize=(8, 8))
    image_data = nib.load(file_path).get_fdata()
    if image_data.ndim == 4:
        image_data = image_data[:, :, :, 1]

    plt.subplot(1, 2, 1)
    plt.imshow(np.fliplr(np.rot90(image_data, k=3)), cmap='gray')
    plt.title("Original NIFTI Image")

    plt.subplot(1, 2, 2)
    visualization_image = Image.fromarray(np.fliplr(np.rot90(visualization)))
    visualization_image = ImageOps.expand(visualization_image, border=(10, 10, 10, 30), fill='black')
    draw = ImageDraw.Draw(visualization_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 24)
    draw.text((10, 266), prediction_tag, fill="white", font=font)
    plt.imshow(visualization_image)
    plt.title("GradCAM Visualization")

    plt.tight_layout()
    plt.show()

    print(f"Predicted Tag: {prediction_tag}")
    print(f"Prediction Confidence: {predictions.flat[predictions.argmax()]:.4f}")

def visualize_prediction_side_colab(file_path):
    """
    Visualize predictions for a NIFTI image file using the side model.

    Args:
        file_path (str): Path to the NIFTI image file.
    """
    print('----------------------------------------- Body Side -----------------------------------------')

    predictions, visualization = predict_from_file_path(file_path, 'XraySorterAI/models/model_side')
    prediction_tag = TAGS_SIDE[predictions.argmax()]

    plt.figure(figsize=(8, 8))
    image_data = nib.load(file_path).get_fdata()
    if image_data.ndim == 4:
        image_data = image_data[:, :, :, 1]

    plt.subplot(1, 2, 1)
    plt.imshow(np.fliplr(np.rot90(image_data, k=3)), cmap='gray')
    plt.title("Original NIFTI Image")

    plt.subplot(1, 2, 2)
    visualization_image = Image.fromarray(np.fliplr(np.rot90(visualization)))
    visualization_image = ImageOps.expand(visualization_image, border=(10, 10, 10, 30), fill='black')
    draw = ImageDraw.Draw(visualization_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 24)
    draw.text((10, 266), prediction_tag, fill="white", font=font)
    plt.imshow(visualization_image)
    plt.title("GradCAM Visualization")

    plt.tight_layout()
    plt.show()
    print(f"Predicted Tag: {prediction_tag}")
    print(f"Prediction Confidence: {predictions.flat[predictions.argmax()]:.4f}")