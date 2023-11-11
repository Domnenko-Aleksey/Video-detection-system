import json
import os
import numpy as np
import pandas as pd
import cv2
import supervision as sv
import time

# from process_prediction import process_prediction
# import ffmpeg



#################################
##### slow object parametrs #####
CLASS_NAMES = ["Vendor"]
ID_LIMIT = 10

IOU_THRESH = 0.5

# Отслеживаемые объекты:
static_objects = dict() # координаты
valid_idx = dict()      # сколько времени отслеживается (счетчик времени жизни)
detected_idx = dict()   # сколько времени стоит на месте

# болванка для корректной работы
static_objects[0] = np.array([0, 0, 0, 0])
valid_idx[0] = 0
detected_idx[0] = 0
#################################


def IOU(box1: np.ndarray, box2: np.ndarray):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return (base_mat * intersect_ratio)[0]




def process_prediction(pred, DETECTED_LIM):
    '''
    Функция обрабатывает и размечает полученные кадры.
    Кроме того идёт поиск медленных или неподвижных объектов, распознанных как торговец
    '''
    frame = pred.orig_img
    sv_preds = sv.Detections.from_ultralytics(pred)    #  передаем предсказание в библиотеку супервижн

    # получаем из супервижна координаты боксов, уверенность и лейблы
    bboxes_xyxy = sv_preds.xyxy
    confidence = sv_preds.confidence
    labels = sv_preds.class_id
    # собираем все вместе в формате (x1, y1, x2, y2, confidense, class_id)
    final_prediction = np.concatenate([bboxes_xyxy, np.expand_dims(confidence, 1), np.expand_dims(labels, 1)], axis=1)
    
    # проходим по всем предсказаниям
    for pred in final_prediction:

        # получамем координаты бокса
        class_id = int(pred[5])
        pred_confidence = round(pred[4], 2)
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        
        # добавляем текст: имя класса - уверенность
        text = str(CLASS_NAMES[class_id]) + ": " + str(pred_confidence)
        
        # рисуем бокс и подпись к нему
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 50), (x1 + len(text) * 33, y1), (0, 0, 255), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)


    # Проверка, что найденный торговец медленно движется или стоит на месте
    # Если нет отслеживаемых объектов - добавляем болванку, чтобы код не рухнул
    if len(valid_idx) == 0:
        static_objects[0] = np.array([0, 0, 0, 0])
        valid_idx[0] = 0
        detected_idx[0] = 0
    # смотрим какой индекс не занят
    max_vendor_id = max(valid_idx.keys())    
    valid_idx_keys = list(valid_idx.keys())

    # проходим по всем отслеживаемым объектам
    for vendor_id in valid_idx_keys:
        # если чего то не хватает - заполняем нулями
        detected_idx.setdefault(vendor_id, 0)
        valid_idx.setdefault(vendor_id, 0)

        # бокс, по которому будет отслеживаться ИОУ
        finded_box = np.array([0, 0, 0, 0])
        box_b = static_objects[vendor_id][:4]

        # проходим по всем найденным моделью объектам
        for pred in final_prediction:
            # бокс, по которому будет отслеживаться ИОУ
            box_a = pred[:4] 
            # если ИОУ достаточен - сохраняем бокс в отдельную переменную
            if IOU(box_a, box_b) > IOU_THRESH:
                finded_box = box_a
                
        # Если ИОУ достаточек - 
        if IOU(finded_box, box_b) > IOU_THRESH:
            # print("DETECTED")
            # обновляем положение бокса, относительно перемещений отслеживаемого объекта
            new_box = np.mean((finded_box, box_b), axis=0)
            static_objects[vendor_id] = new_box
            
            # фиксируем, что объект медленно движется и включаем счетчик предупреждения            
            detected_idx.setdefault(vendor_id, 0)
            detected_idx[vendor_id] += 1

        # Если ИОУ недостаточен - 
        else:
            # добавляем неизвестныe объекты в список отслеживаемых
            for pred in final_prediction:
                static_objects[max_vendor_id + 1] = pred[:4]
                valid_idx[max_vendor_id + 1] = 0
            
            # обновляем счетчик времени жизни детекции
            valid_idx.setdefault(vendor_id, 0)
            valid_idx[vendor_id] += 1

        # Если объект стоит на месте долго - отмечаем красным квадратом
        if detected_idx[vendor_id] > DETECTED_LIM:
            text = "WARNING"
            x1, y1, x2, y2 = static_objects[vendor_id].astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 8)
            cv2.rectangle(frame, (x1 - 1, y1 - 45 - 50), (x1 + len(text) * 45, y1 - 50), (0, 0, 255), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 58), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)

        # Если время жизни отслеживаемого объекта истекло - удаляем его
        if valid_idx[vendor_id] > ID_LIMIT:
            del valid_idx[vendor_id]
            del static_objects[vendor_id]
            del detected_idx[vendor_id]


    return frame




def model_predict_ajax(CORE):
    print('PATH: /model_predict_ajax/model_predict_ajax.py')

    print('POST:', CORE.post)

    DETECTED_TIME = int(CORE.post['time_treshold'])
    DETECTED_LIM = DETECTED_TIME * 25
    predict_list = CORE.model.predict('files/video.mp4', device=0)


    width = 640
    hieght = 480
    channel = 3
    
    fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    video = cv2.VideoWriter('files/predict.webm', fourcc, float(fps), (width, hieght))


    for r in predict_list:
        # im_array = r.plot()

        # обрабатываем кадр, 
        im_array = process_prediction(r, DETECTED_LIM)

        frame = cv2.resize(im_array, (640, 480))
        print('RESIZE:', frame.shape)

        video.write(frame)
    video.release()

    answer = {'answer': 'success'}
    return {'ajax': json.dumps(answer)}
