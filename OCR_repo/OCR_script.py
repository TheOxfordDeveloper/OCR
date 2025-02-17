import os
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Pre-processing methods:
def denoise(image):
    """Reduces noise in the image using a median blur filter."""
    return cv2.medianBlur(image, 5)

def sharpen(image):
    """Sharpens the image using a Laplacian filter."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# File containing paths to images
image_list_file = '/Users/theoxforddevelopr/Desktop/OCR_repo/tests/data/images.txt'
# Replace with your actual file path

# Read image paths from the text file
with open(image_list_file, 'r') as file:
    image_paths = file.readlines()

# Process each image
for image_path in image_paths:
    image_path = image_path.strip()  # Remove newline characters and spaces
    if not image_path:
        continue  # Skip empty lines

    # Load the image using OpenCV
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        print(f"Error loading image: {image_path}")
        continue

    # Convert to grayscale
    # (retains intensity calculated as a weighted combination of the RGB values) 
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/greyscale.jpg', img_gray)

    # image binarisation 
    # Converts an image into a binary image with only two intensity levels: 0 (black) and 255 (white). 
    # This is done using a thresholding operation: 
    # If a pixel's intensity is greater than the threshold, it is set to white (255).
    # If it's less than or equal to the threshold, it is set to black (0) - so want black text to fall below 
    # threshold to stay black once binarised 

    # input has to be greyscale image - THRESHOLD FUNCTION DOES BINARISATION SO COMMENT THIS OUT 

    # _, img_binarised = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # save binarised image 
    # cv2.imwrite('/Users/chloefairhurst/Desktop/OCR_1/pytesseract/tests/preprocessed_images/binarised_image.jpg', img_denoised)


    # Apply sharpening
    img_sharpened = sharpen(img_gray)

    # save sharpened image 
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/sharpened.jpg', img_sharpened)


    # invert colours (white text black background - easier to detect text)
    img_inverted = cv2.bitwise_not(img_sharpened) 

    # save inverted image 
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/inverted.jpg', img_inverted)



    # add thresholding to separate objects of interest from background / aka binarise the image 

    img_threshold = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # save threshold image 
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/threshold.jpg', img_threshold)

  # Apply noise reduction
    img_denoised = denoise(img_threshold)

    # save the de-noised image 
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/denoised.jpg', img_denoised)

    # erode the text to make it thinner as it is a bit thick and characters look like blobs 
    #def thin_font(image):
    #    import numpy as np
    #    image = cv2.bitwise_not(image)
    #    kernel = np.ones((3, 3), np.uint8)
    #    image = cv2.erode(image, kernel, iterations=1)
       # image = cv2.erode(image, np.ones((1,1)))
   #     image = cv2.bitwise_not(image)
    #    return (image)


    #img_eroded = thin_font(img_denoised)
   #cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/eroded.jpg', img_eroded)

# for some reason the dilation function makes the font thinner 
    def thick_font(image):
        import numpy as np
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    img_dilated = thick_font(img_threshold)
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/dilated.jpg', img_dilated)


    
   # Find contours
   # items = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # contours = items[0] if len(items) == 2 else items[1]

    # Create a blank image to draw contours on
    # img_contoured = np.zeros_like(img_threshold)

    # Draw all contours on the blank image
    # cv2.drawContours(img_contoured, contours, -1, (255, 255, 255), 1)  # (255, 255, 255) for white contours

    
    # save contoured image 
    # cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/contoured.jpg', img_contoured)


    # Convert back to RGB for Tesseract
    img_rgb1 = cv2.cvtColor(img_dilated, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('/Users/theoxforddeveloper/Desktop/OCR_repo/tests/preprocessed_images/rgb.jpg', img_rgb1)

    # Perform OCR
    extracted_text = pytesseract.image_to_string(img_rgb1, lang='eng')

    # Define the output CSV file path
    output_csv = os.path.splitext(image_path)[0] + '.csv'  # Saves as <image_name>.csv

    # Write the extracted text to the CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Extracted Text'])  # Header
        csvwriter.writerow([extracted_text])   # Text

    print(f"Text from {image_path} saved to {output_csv}")







