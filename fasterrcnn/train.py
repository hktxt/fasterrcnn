from .model import vgg16


fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)