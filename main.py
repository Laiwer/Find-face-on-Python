import face_recognition
from moviepy.editor import VideoFileClip
import cv2
import os # используется только для удаления временного изображения


def detect_face_on_video(name_videofile:str, with_sound:bool = False, quality_detect:int = 1, model:str = "hog", with_blur:bool = False, increase_area_blur = 0, degree_blur:int = 100) -> None:
    """
    Parameters
    ------------
    name_videofile (required):
        The name of the video file. You only need to enter the file name, without the extension, since only the MP4 format is supported.
    with_sound (optional):
        If this parameter is True, then audio will be loaded to the video, otherwise the video will be without sound.
    quality_detect (optional):
        This is the number that determines the size of the face found in the video. The bigger it is, the more small faces the program finds. The optimal range is: 1 - 5.
    model (optional):
        It is better to read about it in the official documentation: https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations
    with_blur (optional):
        If this parameter is set to True, then when a face is found on the video, it will blur, otherwise the face will simply be highlighted with a blue square without blurring.
    increase_area_blur (optional):
        Works only with blur. This is the number by which the blur area of the found face will be increased. Measured in pixels.
    degree_blur (optional):
        Works only with blur. The level of blurring of the face, which is measured as a percentage (from 0% to 100%).
    """
    blur = "blur" if with_blur else "" # Используется только для названия файла
    # Импортируем видео
    cap = cv2.VideoCapture(f"{name_videofile}.mp4")
    # Этот импорт видео используется для точной передачи FPS и размеров оригинального видео
    video = VideoFileClip(f"{name_videofile}.mp4")
    # Создаётся новое видео, которое мы будем заполнять обработаными изображениями
    clip = cv2.VideoWriter(f"new_{quality_detect}_{blur}_{name_videofile}.mp4", cv2.VideoWriter.fourcc(*"mp4v"), video.fps, video.size)
    
    # print(video.fps, video.size) # вывод в консоль FPS и размеры видео

    # бесконечный цикл до тех пор, пока не обработаем всё видео
    while True:
        # считываем одно изображение из видео
        success, img = cap.read()
        # если считалось без ошибок
        if success == True:
            # конвертируем в чёрно-белое изображение (не до конца понял зачем, что мешает использовать цветное, наверное так точнее)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # временно сохраняем ЧБ изображение
            cv2.imwrite(f"tmp_{name_videofile}.jpg", img_gray)
            # загружаем изображение в виде NDArray
            image = face_recognition.load_image_file(f"tmp_{name_videofile}.jpg", mode="L")
            # находим координаты лица (лиц) на изображении (подробнее про параметры см. в описании функции или на Github: https://github.com/Laiwer/Find-face-on-Python)
            faces = face_recognition.face_locations(image, number_of_times_to_upsample=quality_detect, model=model)
            # получаем координаты каждого лица
            for (top, right, bottom, left) in faces:
                if with_blur:
                    # следующие переменные нужны только для сокращения их названия
                    num = increase_area_blur
                    x, y, w, h = left, top, right, bottom
                    try:
                        # мы используем срез области изображения по координатам, чтобы размыть отдельно эту область изображения
                        img[y-num:h+num, x-num:w+num] = cv2.blur(img[y-num:h+num, x-num:w+num], (degree_blur, degree_blur))
                    except cv2.error:
                        # если найденое лицо с параметром 'increase_area_blur' выходит за рамки изображения, то область размытия не увеличивается и остаётся преждней
                        img[y:h, x:w] = cv2.blur(img[y:h, x:w], (degree_blur, degree_blur))
                elif not with_blur:
                    # рисуем синий квадрат вокруг найденного лица
                    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            # сохраняем изображение в видео
            clip.write(img)
        # если возникла какая-то ошибка, то прерываем обработку видео
        else:
            break
    # удаляем временное ЧБ изображение
    os.remove(f"tmp_{name_videofile}.jpg")
    # Закрываем и сохраняем видео
    cap.release()
    clip.release()

    if with_sound:
        # создаём новый объект видео без аудио
        new_video = VideoFileClip(f"new_{quality_detect}_{blur}_{name_videofile}.mp4")
        # и присваиваем этому объекту аудио оригинального видео
        new_video.audio = video.audio
        # удаляем видео без звука (перезаписать по названию не получается, поэтому пересоздаём)
        os.remove(f"new_{quality_detect}_{blur}_{name_videofile}.mp4")
        # сохраняем видео уже со звуком
        new_video.write_videofile(f"new_{quality_detect}_{blur}_{name_videofile}.mp4")


if __name__ == "__main__":
    detect_face_on_video("people-applauds", quality_detect=1, with_blur=True, degree_blur=30, increase_area_blur=15)