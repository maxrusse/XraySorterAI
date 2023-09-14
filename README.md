
# A Deep Learning Approach for Projection and Body Side Classification in Musculoskeletal Radiographs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxrusse/XraySorterAI/blob/master/XraySorterAI_Demo.ipynb)


## Brief:
This project introduces a deep learning solution to classify radiographic projection and body side in musculoskeletal radiographs. Utilizing the Xception architecture, the models achieve high accuracy, offering potential improvements in radiologic workflow and patient care.

## ⚠️ Important Disclaimer:
Usage Rights and Privacy Concerns: Ensure you have the appropriate rights to use any X-ray images before uploading or using them with this model. Remove any personally identifiable information from the images. Be aware of the ethical and legal implications related to patient privacy and data protection.

Not for Medical Use: The predictions and visualizations provided by this model are strictly for demonstration and research purposes. They should not be used for clinical or diagnostic purposes and do not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical condition.

## Download and Installation:

- **Clone the GitHub repository**:
```bash
git clone https://github.com/maxrusse/XraySorterAI.git
```

- **Download the Repository**:
  1. Navigate to the main page of the repository at https://github.com/maxrusse/XraySorterAI.
  2. Click "Code", then "Download ZIP".
  3. Extract the ZIP to your desired location.

## Usage:


**Install required libarys**:
- TensorFlow version: 2.13.0
- tf-explain version: 0.3.1
- nibabel version: 4.0.2
- cv2 version: 4.8.0
- numpy version: 1.23.5
- Python version: 3.10.12

## Integration

1. Navigate to the "models" directory to access the pretrained models.
2. Integrate the models as per your requirements, exemplified by the "visualize_prediction" functions in "analyse.py".

## Quick Start with Google Colab:

For a hands-on demo, [click here](https://colab.research.google.com/github/maxrusse/XraySorterAI/blob/master/XraySorterAI_Demo.ipynb)

## License:

This project is licensed under the MIT License. Please refer to the LICENSE file in the repository for more details.


[![DOI](https://zenodo.org/badge/691215990.svg)](https://zenodo.org/badge/latestdoi/691215990)
