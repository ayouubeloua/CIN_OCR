# Importer les bibliothéques
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import re
import copy
import shutil
import face_recognition
import argparse
import traceback
from flask import Flask, request, jsonify
import os
import normal_ocr
from werkzeug.utils import secure_filename

################# Do Magic CIn ##############################################

#Determiner specification dictio 

CIN_Regex = r'[a-zA-Z]{1,2}[0-9]{5,6}'
DATE_Regex = r'[0-9]{2}(\s|\.)[0-9]{2}(\s|\.)[0-9]{4}'
KEYWORDS = ['royaume', 'carte', 'nationale', 'identite', 'maroc']
NAME_Regex = r'([A-Z]+|[A-Z]+(\s|-){1}[A-Z]{3,20})'


def do_magic(path):
    filename, file_extension = os.path.splitext(path)
    #extraire l'image à partir d'un pdf
    if file_extension == '.pdf':
        pages = convert_from_path(path)
        #extraire l'image à partir d'une image

        pages[1].save(filename + '.jpg')
        return predict(filename + '.jpg')

    if file_extension == '.jpg':
        return predict(path)

    if file_extension == '.png':
        return predict(path)

# image preprocessing 
def preprocess_image(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 28)
    kernel = np.ones((1, 1), np.uint8)
    #Appliquer la morphologie
    dilation = cv2.dilate(filtered, kernel, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    or_image = cv2.bitwise_or(img, closing)
    or_image = or_image.astype(np.uint8)
    img_name = 'CIN1.png'##path to image 
    or_image = cv2.imwrite(img_name, or_image)
    img = cv2.imread(img_name)
    #image au niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    filtered_img = 'filterImg.png'
    cv2.imwrite(filtered_img, th1)
    return filtered_img

# nettoyer le contenu 

def cleanse_content(content):
    content_list = copy.deepcopy(content)
    for e in content_list:
        if e == '' or e.lower() == e:
            content.remove(e)
        else:
            for k in KEYWORDS:
                if k in e.lower():
                    try:
                        content.remove(e)
                    except:
                        pass
    return content

# Verifier les majuscules

def check_for_uppercase(input):
    counter = 0
    before_value = False
    for e in input:
        if len(re.findall(r'[A-Z]', e)) > 0:
            if before_value:
                counter += 1
            before_value = True
        else:
            if before_value:
                before_value = False
                counter = 0
        if counter == 3:
            return True
    return False

#fonction de prediction 
def predict(filtered_img):
    # utiliser pytesseract pour identifier le texte dans l'image traité 
    content = pytesseract.image_to_string(Image.open(preprocess_image(filtered_img))).split('\n')
    #champs à prédir 
    date = ''
    cin = ''
    last_name = ''
    first_name = ''
    index_of_datefound = 0
    for e in content:
        if re.search(CIN_Regex, e):
            cin = re.search(CIN_Regex, e).group()
        if re.search(DATE_Regex, e) and date == '':
            now = datetime.datetime.now()
            date_found = re.search(DATE_Regex, e).group()
            index_of_datefound = content.index(e)
            try:
                if int(date_found[len(date_found) - 4:]) < now.year:
                    date = date_found[:2] + '/' + date_found[3:5] + '/' + date_found[len(date_found) - 4:]
            except:
                pass
    if index_of_datefound > 0:
        content = content[0: index_of_datefound]
    content = cleanse_content(content)
    printed_content = copy.deepcopy(content)
    last_element = content.pop()
    try:
        last_name = re.search(NAME_Regex, last_element).group()
    except:
        pass
    for i in range(len(content)):
        e = content[len(content) - i - 1]
        if len(e.strip().replace(' ', '')) > 3 and check_for_uppercase(e.strip().replace(' ', '')):
            try:
                first_name = re.search(NAME_Regex, e).group()
                break
            except:
                pass
    result = {'First Name': first_name, 'CIN': cin, 'Birth Date': date, 'Last Name': last_name}
    return result
################# API ##############################################
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, instance_relative_config=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
HOST_PORT = 5006


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[len(filename.rsplit('.', 1)) - 1].lower() in ALLOWED_EXTENSIONS


@app.route('/magic_link', methods=['POST'])
def magic_link():
    if request.method != 'POST':
        return 'Forbidden'
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = file.filename.split('.')[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'card.' + extension))
            return jsonify(normal_ocr.do_magic('uploads/card.' + extension))


def main():
    parser = argparse.ArgumentParser(description='Id Card Recognition')
    parser.add_argument('-p', '--port',type=int, default=HOST_PORT, help='port number')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    if args.debug:
        app.debug = True
    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()

################# Face Detection ##############################################
image_path = 'C:/Users/hp/Desktop/HED/Face_Matching/img'
all_image_path = 'C:/Users/hp/Desktop/HED/Face_Matching/Images'
move_path = 'C:/Users/hp/Desktop/HED/Face_Matching/match'
all_image_path2 = 'C:/Users/hp/Desktop/HED/Face_Matching/CINs'
def func_detect():
	#chemin vers le model de haarcascade 
    face_cascade = cv2.CascadeClassifier('C:/Users/hp/Desktop/HED/Face_Matching/har_model/haarcascade_frontalface_default.xml')
    
    #itérer sur le répértoire cible et scrapper le visage dans les images 

    for i in os.listdir(all_image_path):
            img = cv2.imread(os.path.join(all_image_path, i))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y-100),(x+w,y+h+100),(255,0,0),2)
                rect = np.array([[x,y-100], [x+w,y-100],  [x, y+h+100], [x+w,y+h+100]])
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                rect = np.array([[x,y], [x+w,y],  [x, y+h], [x+w,y+h]])
                #tracer le rectangle bleu
                print(rect)
                roi_gray = gray[y:y+h+100,x:x+w]
                roi_color =img[y:y+h+100,x:x+w]
                rectangle = np.zeros((4,2),dtype = "float32")
                rect = np.vstack(rect).squeeze()
                s=rect.sum(axis = 1)
                rectangle[0] = rect[np.argmin(s)]
                rectangle[2] = rect[np.argmax(s)]
    
                diff = np.diff(rect,axis = 1)
                rectangle[1] = rect[np.argmin(diff)]
                rectangle[3] = rect[np.argmax(diff)]
    
                (tl, tr, br, bl) = rectangle
    
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
    
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([
                    	[0, 0],
                    	[maxWidth - 1, 0],
                    	[maxWidth - 1, maxHeight - 1],
                    	[0, maxHeight - 1]], dtype = "float32")        
                M = cv2.getPerspectiveTransform(rectangle , dst)
                warped = cv2.warpPerspective(img , M ,(maxWidth , maxHeight))
                h,w,f = warped.shape
                warped = cv2.resize(warped,(int(w*2),int(h*2)))
                warped = cv2.resize(warped, (300,300))
                #
                cv2.imwrite("Haar_img/detected"+i+".png",warped)
                
    return rect , warped
            
######################Face Matching############################################             
def func_match(known_image_loc, unknown_image_loc, copy_image_loc, size , tolerance ):
        
      # Ouvrir tous les images 
    for i in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size,size))
      
        print(image.shape)
     
      # renvoyer une liste  des encodages pour toutes les faces de l'image originale 
        face_image_encodings = face_recognition.face_encodings(image)    
        for single_encoding in face_image_encodings:
    
        # itérer sur toutes les images dans le répertoire cible
         for x in os.listdir(all_image_path2):
    
          new_image = cv2.imread(os.path.join(all_image_path2, x))
          new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
          new_image = cv2.resize(new_image, (size,size))
    
          # renvoyer une liste des encodages des visages dans le répertoire cible 
          new_face_image_encodings = face_recognition.face_encodings(new_image)
    
          for each_encoding in new_face_image_encodings:
    
                # comparer les encodages et copier la cin match .
            if face_recognition.compare_faces([single_encoding], each_encoding, tolerance = tolerance)[0]:
              
              shutil.copyfile(os.path.join(all_image_path2, x), os.path.join(move_path, x))
              print('Match => Copié ')
              break
    
            else:
                print('No Matching)')
            
rect = func_detect()

func_match(image_path, all_image_path2, move_path, size = 300, tolerance = 0.6)