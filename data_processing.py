import re
import os
import cv2
import glob
import shutil
import pytesseract
import pandas as pd
from PIL import Image

def extracao_dados(path_videos: str) -> pd.DataFrame:
    """A funcao extracao_dados importa os videos em formato mp4 e extrai a temperatura
    maxima identificada pela camera naquele video. A funcao importa os videos e separa
    o video em frames. Podemos escolher a quantidade de frames por segundo a serem
    extraidos. Tendo os frames dos videos, extraimos todas as temperaturas e, por
    fim, filtramos apenas a temperatura maxima por video.
    
    inputs:
        
        path_videos: Caminho para a pasta onde as filmagens das cameras estao salvas.
        
    returns:
        
        df: Dataframe com o dado de temperatura maxima por video.
        
    """
    # Pega o caminho de todos os videos utilizados como inputs
    
    videos_list = glob.glob(path_videos+"*.mp4")
    for i in range(len(videos_list)):
        videos_list[i] = videos_list[i].replace("\\", "/")
    
    temp_max = {}
    
    # Loop que faz a separacao dos videos em frames e em seguida extrai o valor
    # de temperatura em cada frame.

    for video in videos_list:
        
        cap = cv2.VideoCapture(video)

        frame_rate = 1  # Quantidade de frames extraido do video por segundo
        frame_count = 0

        # Nome do video
        
        video_name = os.path.splitext(os.path.basename(video))[0]

        # Create an output folder with a name corresponding to the video
        output_directory = os.getenv("PATH_VIDEOS")+video_name+"_frames"
        os.makedirs(output_directory, exist_ok=True)

        temp = []

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Only extract frames at the desired frame rate
            if frame_count % int(cap.get(5) / frame_rate) == 0:
                output_file = f"{output_directory}/frame_{frame_count}.jpg"
                cv2.imwrite(output_file, frame)
                print(f"Frame {frame_count} has been extracted and saved as {output_file}")
                im = Image.open(output_file)
                width, height = im.size
                crop_rectangle = (0, 0, width/5, height/8)
                numero = im.crop(crop_rectangle)
                numero.save(output_file)
                
                pytesseract.pytesseract.tesseract_cmd = os.getenv("PATH_TESSERACT")

                image = cv2.imread(output_file, 0)
                thresh = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO)[1]
                data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')

                data = re.sub("[^\d\.]", "", data)
                
                if len(data) >= 1 and data[-1] == ".":
                    data = data[:-1]
                
                if data == "" or len(data) >= 5:
                    pass
                else:
                    try:
                        temp.append(float(data))
                        if float(data) >=100:
                            temp.pop()
                    except:
                        pass
                
        temp_max[video_name] = max(temp)
        shutil.rmtree(output_directory)
        
    df = pd.DataFrame(temp_max.items(), columns=['video', 'max_temp'])
    
    return df