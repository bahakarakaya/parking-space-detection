import os
import cv2
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not


# image1 ve image2 arasındaki farkı hesaplayan fonksiyon 
def calc_diff(im1, im2): # resimdeki pixellerin ortalama sayısı. im1 ortalamsı - im2 ortalaması. Rough estimation (kaba tahmin)
    return np.abs(np.mean(im1) - np.mean(im2))

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

mask_path = './mask_1920_1080.png'
video_path = "./samples/parking_1920_1080_loop.mp4"

mask = cv2.imread(mask_path, 0) #0 gray scale için

#videonun okunması
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
"""
    Graf teorisinden temel alan bir tekniktir
    Bu fonksiyon bir binary mask üzerindeki bağlı bileşenleri (connected_components) tespit etmek için kullanılır
    Her bir connected componenti ayrı bir etiketle tanımlar
    Amaç: Maskedeki farklı park yerlerini tanımlamak ve ölçmek için kullanıldı
    
    mask : Beyaz (1) pikseller tespit edilmek istenen nesneler, siyah (0) arka plan
    4 : Komşuluk türü, 4 yönlü komşularıyla bağlantılı olup olmadığı kontrol edilir. alternatifi 8 (çaprazlar dahil)
    cv2.cv32S : Çıktı etiketlerinin (labels) türünü belirler. bu durumda her connected component için bir etiket
                (int) oluşturulacak ve 'int32' formatında olacak.
"""

spots = get_parking_spots_bboxes(connected_components)
#componentlerin kutu içerisinde koordinatlarını döndürür

spots_status = [None for j in spots]
diffs = [None for j in spots] # park yerinde bir şey olup olmadığının tespiti için veri kaydı yapar.

previous_frame = None #her iteration yapıldığında önceki frame bu değişkene kaydedilir

ret = True #videonun henüz bitmediğini gösteren belirteç
step = 30 #kontrol sıklığı, 30 frame'de bir
frame_nmr = 0
while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :] #geçerli park yeri

            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :]) #mevcut ve önceki frame arasındaki farkı hesaplar, kaydeder

        #print([diffs[j] for j in np.argsort(diffs)][::-1])
        """
            np.argsort(dizi) -> çıktı: bir dizinin küçükten büyüğe sıralanmış indekslerini döndürür.
            [diffs[j] for j in np.argsort(diffs)] yukarıdan gelen gelen sıralı indeksleri kullanarak sıralı elemanlardan oluşan yeni bir liste oluşturur.
            [::-1] -> Listeyi ters çevirerek büyükten küçüğe sıralar
        """

    # park yeri durumunu 30 saniyede bir günceller. (video fps 30)
    if frame_nmr % step == 0:

        if previous_frame is None: #sadece ilk frame'de isek. çünkü prev None ise hata verecek
            arr_ = range(len(spots)) # park yeri sayısı aralığında bir diziye eşitler, 
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            #eğer verilen fark değeri 0.4 üstündeyse tüm spotlar üzerinden ilerler. bu farkı histogramlara bakarak belirledi

        for spot_indx in arr_:

            spot = spots[spot_indx]
            
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :] #geçerli park yeri
            spot_status = empty_or_not(spot_crop) #park yeri boş/dolu ?
            
            spots_status[spot_indx] = spot_status #boş/dolu yerleri listeye kaydeder


    if frame_nmr % step == 0: 
        previous_frame = frame.copy() #mevcut frame'i sonrakiyle karşılaştırma yapmak için kaydeder


    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx] #çalışılan indexteki park durumunu kontrol eder
        x1, y1, w, h = spots[spot_idx] #çalışılan indextekki park yerinin koordinatlarını alır

        #boş ise yeşil, dolu ise kırmızı
        if spot_status:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,0,255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0,0,0), -1) #yazı arka planı
    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spots_status)), str(len(spots_status))), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # spots_status 'un toplamını alır ve listenin toplam uzunluğuna böler. "spots_status" boolean olduğu için her True 1 olarak sayılır ve bu değer toplanarak elde edilmiş olur

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL) #açılan pencerenin yeniden boyutlandırılabilmesini sağlar
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

frame_nmr += 1



cap.release()
cv2.destroyAllWindows()