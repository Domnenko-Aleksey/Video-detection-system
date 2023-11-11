import os
import shutil
import json
import numpy as np
import pandas as pd


def video_upload_ajax(CORE):
    print('PATH: /video_upload_ajax/video_upload_ajax.py')

    # predict_video_path = 'files'
    # if os.path.isdir(predict_video_path):
    #     shutil.rmtree(predict_video_path)

    source_video_path = 'files/video.mp4'
    if os.path.isfile(source_video_path):
        os.remove(source_video_path)

    predict_video_path = 'files/predict.webm'
    if os.path.isfile(predict_video_path):
        os.remove(predict_video_path)

    video_file = CORE.post['video_file'].file.read()

    with open('files/video.mp4', 'wb') as f:
        f.write(video_file)
    
    answer = {'answer': 'success'}
    return {'ajax': json.dumps(answer)}