# License Plate Detection App

This project finds license plates in car photos. A user uploads a photo, and the app draws a box around the plate. I used four tools: Roboflow, Google Colab, GitHub and Render.

**Live app:** [paste your Render link here]

## 1. Roboflow (dataset and data preparation)
Roboflow is a website for managing image datasets.
- I searched Roboflow Universe for "license plate detection" and forked a dataset with 2,156 images. Each image already had a box drawn around the plate.
- **Feature engineering:** I added Auto-Orient and Resize (640x640) so all images are the same size. I also added Brightness and Blur augmentation, which created more varied training images. The final version had 5,562 images.
- **Split:** the images were divided into train, validation and test sets.
- I exported the dataset in YOLOv8 format and copied the download code to use in Colab.

## 2. Google Colab (training the model)
Google Colab lets me run Python code online with a free GPU.
- I installed the tools (`ultralytics` and `roboflow`) and downloaded my dataset from Roboflow.
- **Feature extraction and CNN training:** I trained a YOLOv8 model for 20 epochs. YOLOv8 is a CNN, and its layers learn plate features such as edges and shapes by themselves.
- **Visualisation:** I plotted the training curves, the confusion matrix and the precision-recall curve, and tested the model on sample images.
- I downloaded the trained model file, `best.pt`.

**Results:** precision 0.987, recall 0.953, mAP50 0.973.

## 3. GitHub (storing the code)
GitHub is a website for storing project files online.
- I created a repository called `plate-app`.
- I uploaded three files: `app.py` (the Flask web app), `best.pt` (the trained model) and `requirements.txt` (the list of packages the app needs).
- GitHub gives Render a place to read my files from.

## 4. Render (putting the app online)
Render is a hosting website that runs my app so anyone can open it with a link.
- I connected Render to my GitHub repository.
- I set the build command to `pip install -r requirements.txt` and the start command to `gunicorn app:app --timeout 120`.
- I chose the Free plan. Render built the app and gave me a public link.
- On the free plan, the app goes to sleep when idle, so the first visit may take up to a minute.

## How the app works
1. The user opens the Render link and uploads a car photo.
2. The Flask app (`app.py`) sends the photo to the trained model (`best.pt`).
3. The model finds the plate, and the page shows the photo with a box around it.

LIMITATIONS 
1. The model finds the plate but cannot read the plate number, because that would need OCR.
2. It was trained mostly on foreign plates, so it may be less accurate on Rwandan plates and in blurry, dark or angled photos.
3. The free Render plan sleeps when idle, so the first visit can take up to a minute to load.

## Files in this repository
- `app.py`: the web app
- `best.pt`: the trained model
- `requirements.txt`: the packages to install
