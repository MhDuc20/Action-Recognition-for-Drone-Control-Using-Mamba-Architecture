# Drone Control via Hand Gesture Recognition Using Mamba

This project presents a novel approach for drone control through hand gesture recognition, leveraging MediaPipe for pose extraction and the Mamba architecture for efficient sequence processing.

## Introduction

While foundation models based on the Transformer architecture have achieved state-of-the-art performance in many applications, they suffer from computational inefficiencies when handling long sequences, which is critical for gesture recognition. Recent advances in subquadratic-time models, such as structured state space models (SSMs), have demonstrated promise in addressing this challenge. However, these models often struggle with content-based reasoning in complex modalities like human gestures.

To overcome these limitations, we introduce a selective state space mechanism within the Mamba architecture that adapts the SSM parameters based on the input sequence, enabling dynamic propagation or omission of information to enhance gesture understanding. This approach eliminates the need for attention or MLPlayers, leading to a simplified and efficient neural network architecture. Mamba's parallel processing capabilities allow for fast inference, achieving up to 5× higher throughput compared to Transformers and linear scalability in sequence length.

Our proposed Mamba-SSM model achieves outstanding results in gesture recognition, with a test accuracy of 98.14%. Extensive evaluations in real-world scenarios confirm the robustness and responsiveness of the system, demonstrating its potential for drone control in diverse environments. This research positions Mamba as an effective alternative to traditional Transformer-based architectures, paving the way for hands-free interaction with drones across various applications, including surveillance, delivery, and emergency response.

## Features

* **Accurate Hand Gesture Recognition:** Achieves a test accuracy of 98.14%.
* **Efficient Sequence Processing:** Utilizes the Mamba architecture for long sequence handling.
* **Fast Inference:** Up to 5× higher throughput compared to Transformers.
* **Linear Scalability:** Scales efficiently with increasing sequence length.
* **Intuitive Drone Control:** Enables hands-free drone control via hand gestures.
* **MediaPipe Integration:** Accurate pose extraction.
* **Mamba-SSM:** Selective state space mechanism.

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install MediaPipe:**
    ```bash
    pip install mediapipe
    ```
3.  **Download and Install Mamba:**
    (Add specific Mamba installation instructions here, if necessary)
4.  **Run the Application:**
    ```bash
    python main.py
    ```

## Usage

1.  Ensure the drone is connected and powered on.
2.  Launch the gesture control application.
3.  Perform the predefined hand gestures to control the drone.

## Requirements

* Python 3.x
* MediaPipe
* Mamba Architecture
* (Add any other specific requirements)

## Contributing

We welcome contributions to this project. Please submit a pull request to contribute.

## License

This project is licensed under the [MIT/Apache 2.0/etc.] License.

