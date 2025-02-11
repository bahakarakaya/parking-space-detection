import os
import cv2

from util import get_parking_spots_bboxes, empty_or_not


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

ret = True
step = 30 #kontrol sıklığı, 30 frame'de bir
frame_nmr = 0
while ret:
    ret, frame = cap.read()


    # park yeri durumunu 30 saniyede bir günceller. (video fps 30)
    if frame_nmr % step == 0:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :] #geçerli park yeri
            spot_status = empty_or_not(spot_crop) #park yeri boş/dolu ?
            
            spots_status[spot_idx] = spot_status

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]

        #boş ise yeşil, dolu ise kırmızı
        if spot_status:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,0,255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL) #açılan pencerenin yeniden boyutlandırılabilmesini sağlar
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

frame_nmr += 1



cap.release()
cv2.destroyAllWindows()