# detector/views.py

import os
import cv2
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from .forms import VideoUploadForm
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
from django.views.decorators.csrf import csrf_exempt

# Load the trained model once when the server starts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'models', 'deepfake.h5')
model = load_model(MODEL_PATH)

IMG_SIZE = (299, 299)

def home1(request):
    return render(request,"home1.html")
def home2(request):
    form = VideoUploadForm()
    return render(request, 'newdes.html', {'form': form})

def preprocess_frame(frame):
    """Preprocess a single frame for prediction."""
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def home(request):
    """Render the home page with the upload form."""
    form = VideoUploadForm()
    return render(request, 'detector/home.html', {'form': form})

@csrf_exempt
def upload_video(request):
    """Handle video upload, process it, make prediction, and return the result."""
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            video_path = os.path.join(settings.MEDIA_ROOT, video.name)
            
            # Save the uploaded video
            with open(video_path, 'wb+') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)
            
            # Run deepfake detection on the video
            result = detect_deepfake(video_path)
            
            # Optionally, remove the video after processing
            # os.remove(video_path)
            
            return JsonResponse({'result': result})
        else:
            return JsonResponse({'error': 'Invalid form'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def detect_deepfake(video_path):
    """Process the video, make predictions on frames, and determine if fake or real."""
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    MAX_FRAMES = 30  # Limit to first 30 frames for faster processing

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Preprocess frame
        img = preprocess_frame(frame)
        
        # Predict
        pred = model.predict(img)
        predictions.append(pred[0][0])
    
    cap.release()
    
    if not predictions:
        return 'UNKNOWN'
    
    # Calculate average prediction
    avg_pred = np.mean(predictions)
    if avg_pred>0.5:
        return 'FAKE'
    else:
        return 'REAL'
