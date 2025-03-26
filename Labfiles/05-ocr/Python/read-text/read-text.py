from dotenv import load_dotenv
import os
import time
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import fitz  # PyMuPDF

# Import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Menu for text reading functions
        print('\n1: Use Read API for image (Lincoln.jpg)\n2: Read handwriting (Note.jpg)\n3: Read text from PDF (Document.pdf)\nAny other key to quit\n')
        command = input('Enter a number:')

        if command == '1':
            image_file = os.path.join('images', 'Lincoln.jpg')
            GetTextRead(cv_client, image_file)
        elif command == '2':
            image_file = os.path.join('images', 'Note.jpg')
            GetTextRead(cv_client, image_file)
        elif command == '3':
            pdf_file = os.path.join('documents', 'Document.pdf')
            ExtractTextFromPDF(cv_client, pdf_file)
        else:
            print("Exiting...")

    except Exception as ex:
        print(f"Error in main: {ex}")

def GetTextRead(cv_client, image_file):
    print('\n')
    try:
        with open(image_file, "rb") as f:
            image_data = f.read()

        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        if result.read is not None:
            print("\nText:")
            image = Image.open(image_file)
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for line in result.read.blocks[0].lines:
                print(f"  {line.text}")    
                r = line.bounding_polygon
                bounding_polygon = ((r[0].x, r[0].y), (r[1].x, r[1].y),
                                  (r[2].x, r[2].y), (r[3].x, r[3].y))
                draw.polygon(bounding_polygon, outline=color, width=3)
         
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'text.jpg'
            fig.savefig(outputfile)
            print('\n  Results saved in', outputfile)
        else:
            print("No text was detected in the image.")
    except FileNotFoundError:
        print(f"Error: Image file '{image_file}' not found.")
    except Exception as ex:
        print(f"Error processing image: {ex}")

def ExtractTextFromPDF(cv_client, pdf_file):
    try:
        doc = fitz.open(pdf_file)
        page = doc[0]  # Extract text from the first page
        image = page.get_pixmap()
        img_path = "temp_page.jpg"
        image.save(img_path)
        
        print("Extracting text from PDF...\n")
        GetTextRead(cv_client, img_path)
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_file}' not found.")
    except Exception as ex:
        print(f"Error processing PDF: {ex}")

if __name__ == "__main__":
    main()
