# Система видеодетекции объектов нестационарной незаконной торговли

Веб сервис, позволяющий обнаруживать объекты нестационарной торговли с камер наружного размещения. 

Одной из особенностей уличных торговцев 
и нелегальных рекламщиков является то, что они: 
  - либо стоят на месте
  - либо медленно передвигаются от одного человека к другому

Первая часть модели обнаруживает на видео объекты, 
визуально похожие на уличных торговцев

Вторая часть использует алгоритм, который рассчитывает, стоит
обнаруженный объект на месте или движется, и с какой скоростью

Такой подход помогает отсечь ложные срабатывания на случайных прохожих, которые похожи на торговцев, но не стоят на месте, а движутся по своим делам

За основу модели видеодетекции выбрана YOLOv8s, дообученная на 3х датасетах, два из которых были размечены нами вручную.
