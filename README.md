<h1 align="center">Нахождение лица на видео с помощью Face-Recognition</h1>

<div align="center">

![](https://img.shields.io/badge/python--version-3.9-BDC667?style=plastic)
![](https://img.shields.io/github/languages/code-size/Laiwer/Find-face-on-Python?color=77966D&style=plastic)
![](https://img.shields.io/github/stars/Laiwer/Find-face-on-Python?color=626D58&style=plastic)
![](https://img.shields.io/github/watchers/Laiwer/Find-face-on-Python?color=544343&style=plastic)
![](https://img.shields.io/github/last-commit/Laiwer/Find-face-on-Python?color=56282D&style=plastic)
</div>

<div align=center>

![](result_video/gif/new_3_people-cross-the-road.gif)
</div>

## Содержание

* [Идея](#idea)
* [Установка](#install)
* [Как работает?](#how_work)
* [Разбор кода](#code)
* [Примеры](#examples)

### Исходные видео с лицами людей есть в репозитории в папке "video_with_face", а обработанные видео в форматах .mp4 и .gif есть в папке "result_video"

<h2 id="idea">Идея</h2>

Одним воскрестным вечером, я как обычно изучал новую для себя python библиотеку с большим функционалом. На этот раз это была [MoviePy](https://pypi.org/project/moviepy/). Мне понравилась функция размытия в этом модуле (в дальнейшем я буду использовать размытие из модуля [opencv-python](https://pypi.org/project/opencv-python/)). И мне стало интересно поработать с алгоритмом нахождения лица на изображении и последуйщим его размытием. Конкретных туториалов на эту тему я не нашёл, поэтому делаем всё сами.

Сначала я писал код распознования лица с помощью [opencv-python](https://pypi.org/project/opencv-python/), но подумал что нужно более точное нахождение лица на изображении и поэтому решил использовать именно этот python модуль для дальнейшей работы.



<h2 id="install">Установка</h2>
В проекте используется три библиотеки:

``` python
import face_recognition
from moviepy.editor import VideoFileClip
import cv2
...
```
**Face_Recognition** используется для нахождения лица на видео (изображении). [Pypi](https://pypi.org/project/face-recognition/), [Anaconda](https://anaconda.org/conda-forge/face_recognition)

**Moviepy** используется только для того, чтобы при создании нового видео передать такие же параметры FPS и размера картинки как в оригинальном видео. [Pypi](https://pypi.org/project/face-recognition/), [Anaconda](https://anaconda.org/conda-forge/moviepy)

**OpenCV** используется для обработки видео (изображения) и в данном случае для размытия найденого лица. [Pypi](https://pypi.org/project/opencv-python/), [Anaconda](https://anaconda.org/conda-forge/opencv)

Кажется ничего сложного, ввёл в консоль pip и установил нужные пакеты, но на Windows у меня возникли проблемы.
Проблема заключалась в том, что у меня никак не хотел устанавливаться модуль [dlib](https://pypi.org/project/dlib/), который использует [Face-Recognition](https://pypi.org/project/face-recognition/). Я, как нормальный программист, пошёл гуглить проблему и всё что я находил у меня не работало. Я дважды переустанавливал Visual Studio 2019 C++, устанавливал отдельный файл модуля (.whl) и даже, скачая CMake и добавив его в глобальные переменные компьютера, код никак не хотел работать. Но потом я нашёл следующий код:

``` bash
conda install -c conda-forge dlib
```
Удивительно, но о существовании такой прекрасной вещи как **Anaconda** я и подумать не мог. Для этого понадобиться установить CMake и добавить его в переменные([туториал](https://www.youtube.com/watch?v=8_X5Iq9niDE")), установить Anaconda ([туториал](https://www.youtube.com/watch?v=MUZtVEDKXsk)), а также добавить Anaconda в переменные окружения ([туториал](https://stackoverflow.com/questions/47914980/how-to-access-anaconda-command-prompt-in-windows-10-64-bit)). Для скачивания модулей с помощью Anaconda используй сайт для их поиска ([сайт](https://anaconda.org/)).



<h2 id="how_work">Как работает?</h2>

Библиотека [Face-Recognition](https://pypi.org/project/face-recognition/) работает на основе библиотеки [dlib](https://pypi.org/project/dlib/), которая написана на C++. Может скоро я поработаю с библиотекой dlib на Python и сделаю о ней репозиторий с каким-нибудь проектом. Кстати, с определением лиц можно работать напрямую с библиотекой dlib ([пример кода](http://dlib.net/face_detector.py.html)), но с [Face-Recognition](https://pypi.org/project/face-recognition/) более удобнее и меньше кода.

Dlib — это современный набор инструментов C++, содержащий алгоритмы машинного обучения и инструменты для создания сложного программного обеспечения на C++ для решения реальных задач. Он используется как в промышленности, так и в научных кругах в широком спектре областей, включая робототехнику, встраиваемые устройства, мобильные телефоны и большие высокопроизводительные вычислительные среды. Лицензия Dlib с открытым исходным кодом позволяет вам использовать его в любом приложении бесплатно. ([источник](http://dlib.net/))

Обработка видео работает достаточно просто. Библиотека [Face-Recognition](https://pypi.org/project/face-recognition/) поддерживает обработку только изображений. Видео - это изображения, которые меняются с высокой скоростью, например 30 изображений в секунду. Значит чтобы обработать видео, нам нужно обработать каждое изображение, которое содержит видео. На этом и основана функция [detect_face_on_video()](#code)



<h2 id="code">Разбор кода</h2>
Подключив все нужные модули, я приступил к программированию. По сути в самом коде у меня уже добавлены подробные комментарии, поэтому я только поверхностно пройду по функции.

```python
def detect_face_on_video(
    name_videofile: str,
    with_sound: bool = False,
    quality_detect: int = 1,
    model: str = "hog",
    with_blur: bool = False,
    increase_area_blur: int = 0,
    degree_blur: int = 100
) -> None
```
**name_videofile (required):**
    </br>Обязательный. Требуется ввести только имя файла, путь к файлу не подойдёт. Видео должно находиться в том же каталоге, что и скрипт. Использует видео только с расшрирением MP4.</br>
**with_sound (optional):**
    </br>Необязательный. Если этот параметр равен True, тогда видео будет обработано сохранив при этом звук, иначе видео обработается и на выходе вы получите видео без звука.</br>
**quality_detect (optional):**
    </br>Необязательный. Это число, которое определяет рамки области поиска лица на видео, то есть он будет искать самые большие лица на видео, а мелкие пропускать. Чем больше это число, тем более мелкие лица программа будет нахожить на видео, но при этом увеличивается время обработки в несколько раз. Оптимальный диапазон значений: 1 - 5.</br>
**model (optional):**
    </br>Необязательный. Лучше прочитай об этом в офицальной документации: https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations.</br>
**with_blur (optional):**
    </br>Необязательный. Если для этого параметра установлено значение True, то при обнаружении лица на видео оно будет его размывать, иначе лицо будет просто выделено синим квадратом без размытия.</br>
**increase_area_blur (optional):**
    </br>Необязательный. Работает только с включённым размытием. Этот параметр - это число, на которое будет увеличена область размытия найденного лица. Измеряется в пикселях.</br>
**degree_blur (optional):**
    </br>Необязательный. Работает только с включённым размытием. Это параметр уровня размытия лица, то есть с какой силой будет размыто лицо, который измеряется в процентах (от 0% до 100%).</br>
</p>



<h2 id="examples">Примеры</h2>

На **Гиф. 1** поиск лица и выделение его, на **Гиф. 2** поиск лица и размытие его. Качество поиска лица (**quality_detect** из **detect_face_on_video()**) в данном примере равен **1** и обратывалось оно достаточно быстро.

<div align=center>

![](result_video/gif/new_1_people-applauds.gif) **Гиф. 1**

![](result_video/gif/new_1_blur_people-applauds.gif) **Гиф. 2**
</div>

Вот ещё примеры этого же видео, но с качеством поиска лица равным **2**. Разница очень заметна.

<div align=center>

![](result_video/gif/new_2_people-applauds.gif) **Гиф. 3**

![](result_video/gif/new_2_blur_people-applauds.gif) **Гиф. 4**
</div>

Тут уровень качества поиска лица равен **3**. Мой компьютер смог обработать его за ±20 минут.

<div align=center>

![](result_video/gif/new_3_people-applauds.gif) **Гиф. 5**

![](result_video/gif/new_3_blur_people-applauds.gif) **Гиф. 6**
</div>

Можете сравнить все эти видео и понять какой вариант наиболее оптимальный для вас. По моим наблюдениям, я делаю вывод о том, что **Гиф. 3** и **Гиф. 5** схожи и в данном случае **Гиф. 3**, то есть качество поиска лица равное **2**, может хватить. На видео в момент когда люди сидят, можно увидеть что на обоих гифках примерно одинаково определены лица, а когда люди встают, то на **Гиф. 5** определенно всего лишь на 1, 2 лица больше, чем на **Гиф. 3**, но времени на обработку **Гиф. 5** уходит в два раза больше.

Вот ещё пример стокового видео с большим количеством людей, тут я уже оставлю комментарии, просто посмотрите и сами сделайте выводы.

Уровень качества поиска лица равен **1**:

<div align=center>

![](result_video/gif/new_1_people-cross-the-road.gif)
![](result_video/gif/new_1_blur_people-cross-the-road.gif)
</div>

Уровень качества поиска лица равен **2**:

<div align=center>

![](result_video/gif/new_2_people-cross-the-road.gif)
![](result_video/gif/new_2_blur_people-cross-the-road.gif)
</div>

Уровень качества поиска лица равен **3**:

<div align=center>

![](result_video/gif/new_3_people-cross-the-road.gif)
![](result_video/gif/new_3_blur_people-cross-the-road.gif)
</div>