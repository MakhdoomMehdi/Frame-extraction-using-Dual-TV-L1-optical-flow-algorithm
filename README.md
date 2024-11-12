Project Overview
This repository contains code that:

Extracts optical flow frames using the Dual TV-L1 algorithm.
Extracts temporal frames to analyze movement and transitions over time.
Processes videos from two categories: violent and non-violent.
Saves extracted frames into separate folders for easier management and future analysis.
Technologies Used:
Python 3.x
OpenCV for video processing and optical flow extraction
tqdm for progress bars
NumPy for numerical operations
Installation and Setup
Step 1: Clone the Repository
First, clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/violence-detection-frame-extraction.git
cd violence-detection-frame-extraction
Step 2: Install Required Packages
Install the required Python libraries by running:

bash
Copy code
pip install -r requirements.txt
If you don’t have opencv and tqdm installed, you can manually install them using the following commands:

bash
Copy code
pip install opencv-python opencv-contrib-python tqdm numpy
Step 3: Prepare Dataset
Place your video dataset in a folder on your local machine. The dataset should be organized into two main folders:

violent: Contains violent videos.
non-violent: Contains non-violent videos.
The structure should look like this:

markdown
Copy code
Dataset/
    |- Violent/
    |- Non-violent/
Step 4: Running the Code
To run the frame extraction code, execute the Python script:

bash
Copy code
python extract_frames.py
The script will process the videos and save the extracted optical flow frames and temporal frames in corresponding directories.

Folder Structure After Extraction:
After running the script, the extracted frames will be saved in the following structure:

Copy code
Extracted_Frames/
    |- Temporal/
    |- Optical/
Where:

Temporal Frames contain frames that are extracted based on their temporal sequence in the video.
Optical Flow Frames contain frames generated using the Dual TV-L1 optical flow algorithm, which highlights motion between consecutive frames.
How It Works
Optical Flow (Dual TV-L1 Algorithm)
Optical flow techniques estimate the motion between two consecutive frames of video based on the apparent motion of objects. We use the Dual TV-L1 optical flow algorithm, which is known for its robustness in handling large motions, noise, and image variations.

The algorithm computes the flow of pixels between consecutive frames to highlight motion.
These optical flow frames are useful for understanding movement and activity within the video, which is important for violence detection tasks.
Temporal Frames Extraction
Temporal frames are regular frames extracted from videos to capture the sequence of events. These frames help understand the progression of time, which is essential for detecting sudden movements typical in violent videos.

Performance and Time
The time to process videos depends on the video length, resolution, and the algorithm’s computational demand. On average, processing each video may take between 50 seconds to 1 minute, and with 1,000 videos, the total processing time can range from several hours to days.

To improve processing time:

Consider using a lower resolution for videos.
Process videos in parallel (multithreading or multiprocessing).
Use Google Colab for GPU-based acceleration.
Contributing
Feel free to fork this project, submit pull requests, or suggest any improvements! If you encounter any bugs or issues, please open an issue on GitHub.

License
