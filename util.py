import numpy as np
import cv2
import pickle

from skimage.transform import resize


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))


def empty_or_not(spot_bgr):
    flat_data = []

    #işlenebilir formata dönüştürüyoruz
    img_resized = resize(spot_bgr, (15, 15, 3))     # 15x15 image in 3 color channels
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY
    
def get_parking_spots_bboxes(connected_components):
    (total_labels, label_ids, values, centroid) = connected_components
    """
        total_labels : toplam connected components sayısı
        label_ids : ait olduğu her pixel'e ait etiketlerin arrayi
        values : her component için tutulan istatistikler arrayi. ör, kutu koordinatları, boyutu
        centroid : her komponente ait merkezler
    """

    slots = []          #tüm bbox koordinatlarını tutacak liste 
    coef = 1            #gerekirse koordinatları ölçeklemek için

    # her componenti gezinecek döngü
    # 0 genellikle background olduğu için 1'den başlar.
    for i in range(1, total_labels):

        #koordinatlarını çıkart. x1, y1 sol üst. width, height
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])
    
    return slots