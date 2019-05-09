import vgg16


fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=False)