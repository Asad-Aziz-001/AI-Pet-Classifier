# **AI Pet Classifier 🐾**

A modern web application that uses deep learning to classify images of cats and dogs with high accuracy. 

Built with Streamlit and TensorFlow, this app provides an intuitive interface for pet image classification.

# **Live App Link 🔗:** https://ai-pet-classifier-rjgrtbjr6d6keilh6b8vhm.streamlit.app/

# **🌟 Features**

Smart Classification: Upload any image and instantly determine if it's a cat or dog

Confidence Scoring: Get detailed confidence percentages for each prediction

Modern UI: Clean, responsive design with smooth animations

Detailed Insights: View raw prediction scores and model information

Easy Reset: Quickly classify multiple images without page reload

Uncertainty Handling: Intelligently identifies when the model is uncertain about a prediction

# **🛠️ Technologies Used**

Frontend: Streamlit (Python web framework)

Backend: TensorFlow/Keras (Deep learning framework)

Model: MobileNetV2 with transfer learning

Styling: Custom CSS with Google Fonts (Poppins)

Image Processing: PIL (Python Imaging Library)

# **📊 Model Details**

Architecture: MobileNetV2 pre-trained on ImageNet with custom classification head

Training: Transfer learning approach with fine-tuning

Accuracy: ~95% on test dataset

Input Size: 160x160 RGB images

Output: Probability score (0-1) with sigmoid activation

# **🚀 Getting Started**

Prerequisites

Python 3.8 or higher

pip package manager

**Installation**

Install required packages:

pip install -r requirements.txt

# **Run the application:**

streamlit run app.py

Open your web browser and navigate to http://localhost:8501

# **🔧 Model Training**

The model was trained using transfer learning with MobileNetV2 as the base architecture. Key training details:

Dataset: Kaggle Dogs vs. Cats dataset (25,000 images)

Data Augmentation: Random rotations, flips, zoom, and brightness adjustments

# **Training Phases:**

Feature extraction with frozen base layers

Fine-tuning with unfrozen top layers

Optimization: Adam optimizer with learning rate scheduling

Regularization: Dropout (20%) and L2 weight decay

Evaluation: 95% accuracy on held-out test set

# **📝 Usage**

Upload an Image: Click "Choose an image" to select a cat or dog photo

Classify: Click "Classify Image" to run the prediction

View Results: See the prediction with confidence score and progress bar

Explore Details: Expand "See details" for raw scores and model information

Classify Another: Click "Classify Another Image" to reset and try again

# **⚠️ Limitations**

Works best with clear, well-lit images of cats or dogs

May show reduced accuracy for:

Unusual breeds or mixed breeds

Images with multiple animals

Poor quality or heavily filtered images

Non-cat/dog images (will show "Uncertain" prediction)

# **🚀 Future Enhancements**

 Support for additional pet types (birds, rabbits, etc.)
 
 Breed-specific classification
 
 Mobile app version
 
 Batch image processing
 
 Model explanation visualization (Grad-CAM)
 
 Multi-language support
 
# **🤝 Contributing**

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

# **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

# **🙏 Acknowledgments**

Kaggle for the Dogs vs. Cats dataset

TensorFlow team for the deep learning framework

Streamlit team for the amazing web framework

Google Fonts for the Poppins font family

**Developed by ASAD-AZIZ**

If you find this project useful, please consider giving it a ⭐️!
