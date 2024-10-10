# src/run.py

import classifier_descriptores_SVM  # Assuming train.py contains a function to train your model
import classifier_descriptors_FCNN  # Assuming evaluate.py contains a function to evaluate your model
import classifier_descriptors_RF  # Assuming this contains a function for RF
import classifier_representations_CNN_Autoencoder_FCNN  # Assuming this has preprocess logic
import classifier_representations_CNN_FCNN  # Assuming this has preprocess logic
import classifier_representations_RF  # Assuming this has preprocess logic
import classifier_representations_SVM  # Assuming this has preprocess logic
import descriptors_Autoencoder  # Assuming this has preprocess logic

def main():
    # Run preprocessing for each representation
    classifier_representations_CNN_Autoencoder_FCNN.main()  # Call the main function in that module
    classifier_representations_CNN_FCNN.main()  # Call the main function in that module
    classifier_representations_RF.main()  # Call the main function in that module
    classifier_representations_SVM.main()  # Call the main function in that module
    descriptors_Autoencoder.main()  # Call the main function in that module

    # Train models
    classifier_descriptores_SVM.main()  # Call the main function in SVM module
    classifier_descriptors_FCNN.main()  # Call the main function in FCNN module
    classifier_descriptors_RF.main()  # Call the main function in RF module

    # Evaluate models
    classifier_descriptors_FCNN.main()  # Call the main function to evaluate FCNN
    classifier_descriptors_RF.main()  # Call the main function to evaluate RF

if __name__ == "__main__":
    main()
