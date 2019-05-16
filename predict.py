'''Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu'''

# call me by python predict.py --top_k 3
#/path/to/image flowers/test/1/image_06743.jpg
#checkpoint 

import myNetwork as mn
import argparse
import sys

parser = argparse.ArgumentParser(description='Predicts the class of a flower')
parser.add_argument('--top_k', 
                    action = "store", 
                    dest = 'top_k', 
                    default = 3, 
                    type = int,
                    help ='Return top K prediction results')

parser.add_argument('--category_names',
                    action = 'store', 
                    dest = 'category_names',
                    default = 'cat_to_name.json',
                    help = 'Mapping to be used for categories')

parser.add_argument('image_path', 
                    action = 'store',
                    default = 'flowers/test/101/image_07949.jpg',
                    help = 'Used for image path')
 
parser.add_argument('checkpoint_filepath', 
                    action = 'store',
                    default = 'flowers/test/101/image_07949.jpg',
                    help = 'Used for image path')
input_args = parser.parse_args()

print("Let's predict the class of the image {}".format(input_args.image_path))

print('Step 0: Begin processing...')

print('Step 1: Load trained model...')
new_model = mn.load_model_from_checkpoint(input_args.checkpoint_filepath)
print('Step 1: Done')

print('Step 2: Process image...')
mn.process_image(input_args.image_path)
print('Step 2: Done')

print('Step 3: Predict the class of image {} ...'.format(input_args.image_path))
prediction = mn.predict(input_args.image_path, new_model, input_args.category_names, input_args.top_k)
print(prediction)
print('Step 2: Done.')
