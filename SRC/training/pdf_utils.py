from pdf2image import convert_from_bytes
import numpy as np
import cv2

def pdf_to_images(pdf_bytes):

    pages = convert_from_bytes(pdf_bytes)  
    
    cv_images = []
    for page in pages:
 
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        cv_images.append(img)

   
    return cv_images