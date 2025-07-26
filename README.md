# Image-Caption-Generator
This system generates descriptive captions for images using a deep learning architecture that combines:

1.A pretrained DenseNet201 convolutional neural network for image feature extraction
2.An LSTM-based sequence generator for caption generation

The solution includes a user-friendly Streamlit interface for local deployment and testing.

## Dataset 
This project uses the dataset [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) from Kaggle.
The Flickr8k Dataset is a benchmark collection widely used for image captioning tasks. It consists of:
8,000 images selected from six different Flickr groups.
Each image is paired with five distinct captions that clearly describe the main objects and events in the scene.
The dataset features diverse scenes and situations, avoiding well-known people or famous locations.
It provides rich, human-written descriptions ideal for training and evaluating image captioning models.

## Model Architecture
The model consists of two main parts:

**1.Feature Extractor:**
We use a pre-trained DenseNet201 model (without its top classification layer) to extract image features. These features capture rich visual information from input images.

**2.Caption Generator:**
The extracted image features are passed through a dense layer and reshaped, then concatenated with embedded caption sequences.
This combined input is processed by an LSTM network, which generates captions word-by-word.
The output is a softmax layer that predicts the next word in the sequence.

## Streamlit
![Streamlit prediction](assest/Streamlit_pred.png)

## Installation
1.Clone the repository:
```
git clone https://github.com/velish-qubadov/Image-Caption-Generator.git
cd FakeOrRealFace
```
2.(Optional) Create and activate a virtual environment:

Windows:
```
python -m venv venv
venv\Scripts\activate
```
macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```
3.Install dependencies:
```
pip install -r requirements.txt
```
4.Run the Streamlit app:
```
streamlit run app.py
```
