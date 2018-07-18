from flask import Flask, request, Response,redirect,url_for,session,g,make_response,flash,send_file
import jsonpickle
import numpy as np
import cv2
import argparse
import scipy.misc
from PIL import Image 
from utils import *


from model import cyclegan
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='makeup2', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')


tf.set_random_seed(19)
args = parser.parse_args()


# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/', methods=['GET'])
def root():
	response = {'message':'connection success'}
	print(response)
	response_pickled = jsonpickle.encode(response)
	return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/return-file', methods=['GET'])
def return_file():
    return send_file('/home/sunghwankim/Week3Server/test/AtoB_input.jpg', attachment_filename = 'AtoB_input.jpg')    

@app.route('/api/modification', methods=['POST'])
def modification():
  
    r = request
    # convert string of image data to uint8s
    #nparr = np.fromstring(r.data, np.uint8)
    # decode image
    #f = r.files['file']
    file = r.files['upload']
    img = Image.open(file)
    img.save('./datasets/makeup2/testA/input.jpg')
    #f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #file.save(f)
    #f.save(secure_filename(f.filename))

    #img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    '''
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        if model.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        init_op = tf.global_variables_initializer()
        model.sess.run(init_op)
        out_var, in_var = (model.testB, model.test_A)
        img = np.expand_dims(img, axis=0)
        img = np.array(img).astype(np.float32)
        fake_img = model.sess.run(out_var, feed_dict={in_var: img})

    save_images(fake_img, [1, 1], "./output.jpg")
    ## merge images
    '''

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.test(args)
    
    response = {'message': 'image received. size={}x{}'.format(1, 1) }
    print(response)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')
