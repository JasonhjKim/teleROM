import tkinter as tk
import math
import numpy as np
import PIL
from PIL import Image, ImageGrab
import os

from datetime import date
import os
import sys
import time
import scipy.misc
from scipy.misc import imread
from scipy import ndimage, misc
# import imageio

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

import tensorflow.compat.v1 as tf

import numpy as np

import tensorflow.compat.v1 as tf

from nnet.net_factory import pose_net

import math
import pyscreenshot
import numpy as np
from scipy.misc import imresize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.disable_v2_behavior()
tf.reset_default_graph()
tf.initialize_all_variables()
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append(os.path.dirname(__file__) + "/../")
global tic
global toc
# using DeeperCut (https://arxiv.org/abs/1605.03170)
global sess, outputs, inputs, scmap, locref
# ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
# knee -> [0,1,2]root

# elbow -> [3,4,5] 
global joint_array
global scmap_arr, testing_jt
### predict.py file

def prepare_DNN():
    global cfg
    cfg = load_config("demo/pose_cfg.yaml")
    # Compute prediction with the CNN
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size, None, None, 3])
    outputs = pose_net(cfg).test(inputs)
    restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)
    return sess, outputs, inputs

def extract_cnn_output(outputs_np, pairwise_stats = None):
    global cfg
    scmap = outputs_np['part_prob']
    scmap = np.squeeze(scmap)
    locref = None
    pairwise_diff = None
    if cfg.location_refinement:
        locref = np.squeeze(outputs_np['locref'])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    if cfg.pairwise_predict:
        pairwise_diff = np.squeeze(outputs_np['pairwise_pred'])
        shape = pairwise_diff.shape
        pairwise_diff = np.reshape(pairwise_diff, (shape[0], shape[1], -1, 2))
        num_joints = cfg.num_joints
        for pair in pairwise_stats:
            pair_id = (num_joints - 1) * pair[0] + pair[1] - int(pair[0] < pair[1])
            pairwise_diff[:, :, pair_id, 0] *= pairwise_stats[pair]["std"][0]
            pairwise_diff[:, :, pair_id, 0] += pairwise_stats[pair]["mean"][0]
            pairwise_diff[:, :, pair_id, 1] *= pairwise_stats[pair]["std"][1]
            pairwise_diff[:, :, pair_id, 1] += pairwise_stats[pair]["mean"][1]
    return scmap, locref, pairwise_diff


def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1] if offmat is not None else 0
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return np.array(pose)


def argmax_arrows_predict(scmap, offmat, pairwise_diff, stride):
    num_joints = scmap.shape[2]
    arrows = {}
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1] if offmat is not None else 0
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)[::-1]
        for joint_idx_end in range(num_joints):
            if joint_idx_end != joint_idx:
                pair_id = (num_joints - 1) * joint_idx + joint_idx_end - int(joint_idx < joint_idx_end)
                difference = np.array(pairwise_diff[maxloc][pair_id])[::-1] if pairwise_diff is not None else 0
                pos_f8_end = (np.array(maxloc).astype('float') * stride + 0.5 * stride + difference)[::-1]
                arrows[(joint_idx, joint_idx_end)] = (pos_f8, pos_f8_end)

    return arrows

def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


def check_point(cur_x, cur_y, minx, miny, maxx, maxy):
    return minx < cur_x < maxx and miny < cur_y < maxy


def visualize_joints(image, pose):
    marker_size = 8
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = image.shape[1] - 2 * marker_size
    maxy = image.shape[0] - 2 * marker_size
    num_joints = pose.shape[0]

    visim = image.copy()
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for p_idx in range(num_joints):
        cur_x = pose[p_idx, 0]
        cur_y = pose[p_idx, 1]
        if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
            _npcircle(visim,
                      cur_x, cur_y,
                      marker_size,
                      colors[p_idx],
                      0.0)
    return visim

def show_arrows(cfg, img, pose, arrows):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(img)
    a.set_title('Initial Image')


    b = fig.add_subplot(2, 2, 2)
    plt.imshow(img)
    b.set_title('Predicted Pairwise Differences')

    color_opt=['r', 'g', 'b', 'c', 'm', 'y', 'k']
    joint_pairs = [(6, 5), (6, 11), (6, 8), (6, 15), (6, 0)]
    color_legends = []
    for id, joint_pair in enumerate(joint_pairs):
        end_joint_side = ("r " if joint_pair[1] % 2 == 0 else "l ") if joint_pair[1] != 0 else ""
        end_joint_name = end_joint_side + cfg.all_joints_names[int(math.ceil(joint_pair[1] / 2))]
        start = arrows[joint_pair][0]
        end = arrows[joint_pair][1]
        b.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], head_width=3, head_length=6, fc=color_opt[id], ec=color_opt[id], label=end_joint_name)
        color_legend = mpatches.Patch(color=color_opt[id], label=end_joint_name)
        color_legends.append(color_legend)

    plt.legend(handles=color_legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def show_heatmaps(cfg, img, scmap, pose, cmap="jet"):
    global scmap_arr, testing_jt
    interp = "bilinear"
    temp_jts = cfg.all_joints
    temp_jts_names = cfg.all_joints_names
    print(temp_jts_names)
    print(temp_jts)
    if testing_jt == 'RightKnee':
        arr = [0,1,2]
    elif testing_jt == 'LeftKnee':
        arr = [5,4,3]
    elif testing_jt == 'LeftElbow':
        arr = [11,10,9]
    elif testing_jt == 'RightElbow':
        arr = [6,7,8]
    print(testing_jt)

    all_joints = []
    all_joints_names = []
    for i in range(len(arr)):
        all_joints_names.append(temp_jts_names[i])
    # print(all_joints_names)

    scmap_arr = []
    subplot_width = 3

    for pidx, part in enumerate(arr):
        scmap_part = scmap[:, :, part]
        # print(type(scmap_part))
        # print(scmap_part.shape)
        # scmap_part = imresize(scmap_part, 8.0, interp='bicubic')
        scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')
        # plt.figure()
        # plt.imshow(scmap_part)
        # plt.show()
        # c = np.array(scmap_part)
        c_val = np.amax(scmap_part)
        result = np.where(scmap_part==c_val)
        # print(result)
 
        scmap_arr.append(result)
        # print(scmap_part.shape)
    print(scmap_arr)
    # curr_plot = axarr[0, 0]
    # curr_plot.set_title('Pose')
    # curr_plot.axis('off')
    # # curr_plot.imshow(visualize_joints(img, pose))


def classify(image):
    print("classifying...")
    global sess, outputs, inputs, scmap, locref, cfg

    # file_name = "demo/tim_test/0.jpg"
    # image = imread(file_name, mode='RGB')
    
    image = np.array(image)
    image_batch = data_to_input(image)
        
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = extract_cnn_output(outputs_np)
    pose = argmax_pose_predict(scmap, locref, cfg.stride)
    show_heatmaps(cfg, image, scmap, pose)

def take_screenshot():
    print('Screenshotting')
    # d1 = datestamp.strftime("%d-%m-%Y")
    # self.calculate_angle()
    # self.save_string = d1 + '_degrees_'  + str(self.angle) + '_' + str(self.screen_shot_ctr)
    # image_name = os.path.join(str(self.root_path), self.save_string + '.jpg')
    x1_temp = []
    x2_temp = []
    y1_temp = []
    y2_temp = []  
    # print(str(w), str(h))
    # for i in range(len(self.dot_mat)):
    #     x1_temp.append(self.dot_mat[i].x1)
    #     x2_temp.append(self.dot_mat[i].x2)
    #     y1_temp.append(self.dot_mat[i].y1)
    #     y2_temp.append(self.dot_mat[i].y2)     
    # print((min(np.array(x1_temp)),min(np.array(y1_temp)),max(np.array(x2_temp)),max(np.array(y2_temp))))
    
    im_object = pyscreenshot.grab()
    
    # im_object.save('testing.jpg')
    # print("image : " + image_name + " successfully saved" )
    # tic = int(round(time.time() * 1000))
    classify(im_object)
    # print('Screenshotting_2')
    # Extract maximum scoring location from the heatmap, assume 1 person
    # pose = argmax_pose_predict(scmap, locref, cfg.stride)
    # print("pose")
    # print(pose)
    # print("cfg")
    # print(cfg)
    # print("image")
    # print(image)
    # print("scmap")
    # print(scmap)
    # # Visualise
    # visualize.show_heatmaps(cfg, image, scmap, pose)
    # visualize.waitforbuttonpress()

class Dot(object):
    def __init__(self, x1, y1, x2, y2): # constructor
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.center_coordinate = [(x1+x2)/2, (y1+y2)/2]


class draw_output(object):
    ## Declare which jt ##

    global testing_jt

    def __init__(self): # constructor

        self.root = tk.Tk()
        # import pyautogui
        self.height = self.root.winfo_screenheight()
        # height_pass = self.root.winfo_screenheight()
        height_pass = self.root.winfo_screenheight()

        self.width = self.root.winfo_screenwidth()
        width_pass = self.root.winfo_screenwidth()

        # This is my ROOOT dont touch;
        self.root.geometry('%dx%d+%d+%d' % (350, self.height /2 + self.height /3, 0 - 10, 0))
        self.canvas = tk.Canvas(self.root, width=350, height=self.height/2 + self.height/3)
        self.setup()
        self.session_start()
        self.root.mainloop()

        

    def setup(self):
        # Declare variables
        self.scmap = None
        self.movement_ctr = 0
        self.locref = None
        self.keep_screenshotting = True
        self.root_2 = None
        self.root_3 = None
        self.datestamp = date.today()
        self.save_string = None
        self.angle = None
        self.x = None
        self.y = None
        self.pin_joint = '#476042'
        self.array_color = ["red", "yellow", "yellow"]
        self.root_path = os.getcwd()
        self.screen_shot_ctr = 0
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.jt_selected = None
        self.dot_mat = []
        self.dot_arr= None
        self.click_ctr = 1
        self.test = None
        self.distance = None
        self.distance_arr = []
        self.cast_dot_scaling_w = 1/10
        self.cast_dot_scaling_h = 1/30
        self.ind = None
        self.dot_distance = 100
        self.loop_SC = False
        self.is_soapie = False
        self.chosenMovement = []
        self.current_movement = None
        self.done = False
        # Bind keys
        # self.canvas.bind('<Motion>', self.get_mouse_position)
        
        # here is signification
    
    def toggle_screenshot(self,event):
        self.loop_SC = not self.loop_SC # toggles
        self.loop_screenshot()

    def loop_screenshot(self):
        while self.loop_SC:
            self.loop_SC = not self.loop_SC # toggles
            self.canvas_overlay.delete("all")
            take_screenshot()
            
            self.start_dots_from_ui()
            print('Screenshotting')

    def resize(self, event):
        w,h = event.width, event.height
        if (self.initial == False):
            w = 350
            h = self.height/2 + self.height/3
            self.initial = True
        print(w,h)
        self.canvas.config(width=w, height=h)

    def add_one(self, event):
        self.movement_ctr += 1
        # print('on to :' + str(self.chosenMovement[self.movement_ctr]))
        self.canvas_overlay.delete("all")
        
        self.setup_overlay()

    def setup_overlay(self):
        global testing_jt
        if self.movement_ctr < 3:
            i = self.movement_ctr
            print(str(i))
            print(self.chosenMovement)
            self.current_movement =  self.chosenMovement[i]
            print('Current movement: ' + self.current_movement)
            testing_jt = str(self.dir) + str(self.jt_selected)
            testing_jt_print = str(self.dir) + ' ' + str(self.jt_selected) + ' ' + str(self.current_movement)
            self.root_overlay = tk.Tk()
            self.root_overlay.attributes("-alpha", 0.2)
            print(self.width)
            print(self.height)
            self.canvas_overlay = tk.Canvas(self.root_overlay, width=self.width, height=self.height)
            self.make_progress_box(self.root_overlay, 350, 50, self.width/2 - 175, self.height - 100, testing_jt_print)
            self.canvas_overlay.delete("all")
            take_screenshot()
            self.start_dots_from_ui()
            self.canvas_overlay.bind('<B1-Motion>', self.drag_dot)
            # self.canvas_overlay.bind("<Button-3>", self.add_dot) # make a dot
            self.canvas_overlay.bind("<BackSpace>", self.undo) # make a dot
            self.root_overlay.bind('<Escape>', self.close_overlay)
            self.root_overlay.bind('<space>', self.toggle_screenshot)
            self.canvas_overlay.pack()
            self.root_overlay.bind('<Return>', self.add_one)
        else:
            # print(i)
            print('out now')
            self.done = True



    def closeSecondary(self):
        self.secondary.destroy()
        self.setup_overlay()
        

    def session_start(self):
        self.make_option(self.root, 0, 0, 350, 50)
        self.make_button(self.root, 0, 50, 350, 30, callback=self.option_added, text="Add")

    def option_added(self):
        print(self.optionVal.get())
        if self.optionVal.get() == "None":  
            return
        if self.optionVal.get() == "SOAPIE" and self.is_soapie == False:
            print("get in here")
            self.create_soapie(0, 80)
            self.is_soapie = True
        print("Option Clicked")

    def add_rom(self):
        print("open new canvas")
        self.secondary = tk.Tk()
        self.secondary.geometry('%dx%d+%d+%d' % (600, 350,350 - 10, 0))
        canvas = tk.Canvas(self.secondary, width=350, height=self.height/2)
        self.make_body_button(0, 0)

    def make_body_button(self, startX, startY):
        array = ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"]
        for b in array:
            self.make_button_direction(self.secondary, startX, startY, 350, 50, "Left", callback=self.body_button_pressed, text=b)
            startY = startY + 50


    def make_option(self, root, x, y, w, h, *args, **kwargs):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        self.optionVal = tk.StringVar(f)
        self.optionVal.set("None")
        option = tk.OptionMenu(f, self.optionVal, "None", "SOAPIE", "Remark")
        option.pack(fill=tk.BOTH, expand=1, padx=15, pady=10)
        return option
        # option.grid(row=1, columnspan=3, sticky="nesw")
        # option.config(width=175)

    def make_option_with_array(self, root, x, y, w, h, arr, *args, **kwargs):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        mb = tk.Menubutton(f, text="Movement", relief=tk.RAISED, width=w)
        mb.menu = tk.Menu(mb, tearoff=True)
        mb["menu"] = mb.menu
        var = dict()
        for i in range(len(arr)):
            print(i)
            var[i] = tk.IntVar()
            mb.menu.add_checkbutton(label=arr[i], variable=var[i], command= lambda name=arr[i]: self.add_movement_to_arr(name))
        mb.pack(padx=10)
        # self.movement = tk.StringVar(f)
        # self.movement.set(arr[0])
        # option = tk.OptionMenu(f, self.movement, *arr)
        # option.pack(fill=tk.BOTH, expand=1, padx=15, pady=10)
        return mb

    def add_movement_to_arr(self, n):
        print(n)
        foundIndex = self.chosenMovement.index(n) if n in self.chosenMovement else -1
        print(foundIndex)
        if(foundIndex > -1):
            del self.chosenMovement[foundIndex]
        else:
            self.chosenMovement.append(n)
        print(self.chosenMovement)
    
    def make_button_direction(self, root, x, y, w, h, direction, callback, text, *args, **kwargs):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        print("called")
        button = tk.Button(f, text=text, command=lambda: callback(text), width=50)
        button.pack(side = tk.RIGHT if direction == "Right" else tk.LEFT, padx=15)
        return button

    def make_radio_button_type(self, root, x, y, w, h, arr, text):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        w = tk.Label(f, text=text)
        w.pack(side=tk.LEFT)
        self.move_type = tk.StringVar()
        # self.move_type.set()
        for i in range(len(arr)):
            r = tk.Radiobutton(f, text=arr[i], variable=self.move_type, value=arr[i], command= lambda name=arr[i], text=text: self.set_side(name, text))
            r.pack(side = tk.LEFT)
        
    def set_side(self, name, text):
        print("Called")
        if (text == "Side"):
            self.dir = name
            print(self.dir)
        else:
            self.move_type = name
            print(self.move_type)

    def make_radio_button_direction(self, root, x, y, w, h, arr, text):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        w = tk.Label(f, text=text)
        w.pack(side=tk.LEFT)
        self.dir = tk.StringVar()
        # self.dir.set()
        for i in range(len(arr)):
            r = tk.Radiobutton(f, text=arr[i], variable=self.dir, value=arr[i], command= lambda name=arr[i], text=text: self.set_side(name, text))
            r.pack(side = tk.LEFT)


    def make_checkbox_for_joints(self, root, w, h, startX, startY, arr):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        for move in arr:
            self.make_checkbox(f, w, h, startX, startY, move)
            startY = startY + 50

    def make_progress_box(self, root, w, h, x, y, text):
        if isinstance(self.root_3, type(None)):
            self.root_3 = tk.Tk()
        
        self.root_3.geometry('%dx%d+%d+%d'%(w,h,x,y))
        self.root_3.overrideredirect(True)
        self.root_3.attributes("-alpha", 0.8)
        self.make_label(self.root_3, 0, 0, 200, 50, str(text))
    
    def body_button_pressed(self, b):
        self.make_radio_button_direction(self.secondary, 350, 0, 250, 50, ["Left", "Right"], "Side")
        self.make_radio_button_type(self.secondary, 350, 50, 250, 50, ["AROM","PROM"], "Type")
        shoulderArr = ["Flexion", "Extension", "Abduction", "Adduction", "Internal Rotation", "External Rotation"]
        elbowArr = ["Flexion", "Extension"]
        wristArr = ["Flexion", "Extension", "Ulnar Deviation", "Radial Deviation"]
        hipArr = ["Flexion", "Extension", "Abduction", "Adduction", "Internal Rotation", "External Rotation"]
        kneeArr = ["Flexion", "Extension"]
        ankleArr = ["Dorsiflexion", "PlantarFlexion"]
        if (b == "Shoulder"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, shoulderArr)
            self.jt_selected = b
            # self.make_checkbox_for_joints(self.secondary, 350, 150, 250, 50, shoulderArr)
        if (b == "Wrist"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, wristArr)
            self.jt_selected = b
        if (b == "Elbow"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, elbowArr)
            self.jt_selected = b
        if (b == "Hip"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, hipArr)
            self.jt_selected = b
        if (b == "Knee"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, kneeArr)
            self.jt_selected = b
        if (b == "Ankle"):
            self.make_option_with_array(self.secondary, 350, 100, 250, 50, ankleArr)
            self.jt_selected = b
        self.make_button(self.secondary, 350, 150, 250, 50, callback=self.closeSecondary, text="Measure ROM", bg="blue", fg="white")

    def make_button(self, root, x, y, w, h, callback, text, *args, **kwargs):
        f = tk.Frame(root, width=w, height=h)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        print("called")
        button = tk.Button(f, text=text, command=callback, *args, **kwargs)
        button.pack(side=tk.RIGHT, padx=15)
        return button

    def make_label_with_entry(self, root, x, y, w, h, text, *args, **kwargs):
        f = tk.Frame(root, width=w, height=h -30)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        label = tk.Label(f, text=text)
        label.pack(side=tk.LEFT, padx=12)

        f1 = tk.Frame(root, width=w, height=h)
        f1.pack_propagate(0)
        f1.place(x=x, y=y+20)
        var = tk.StringVar(f1)
        var.set("Placeholder text")
        entry = tk.Text(f1)
        entry.pack(fill=tk.BOTH, expand=1, padx=15)

    def create_soapie(self, startX, startY):
        array = ["Subjective", "Objective", "Anaylsis", "Plan", "Intervention", "Evaluation of Effectiveness"]
        for ele in array:
            if ele == "Objective":
                self.make_label_with_entry(self.root, startX, startY, 350, 60, text=ele )
                startY = startY + 80
                self.make_button(self.root, startX, startY, 350, 50, callback=self.add_rom, text="Add ROM")
                startY = startY + 40
            else: 
                self.make_label_with_entry(self.root, startX, startY, 350, 60, text=ele )
                startY = startY + 80
        

    def convert_position(self, nump_arr):
        # print(nump_arr[1][0])
        self.x1, self.y1 = nump_arr[1][0]*6.1 - 1, nump_arr[0][0]*5.9 - 1
        self.x2, self.y2 = nump_arr[1][0]*6.1 + 1, nump_arr[0][0]*5.9 + 1

    def start_dots_from_ui(self):
        global scmap_arr
        # print(str(scmap_arr))
        self.dot_mat = []
        self.convert_position(scmap_arr[1])
        self.add_to_mat(self.x1, self.y1, self.x2, self.y2)
        self.convert_position(scmap_arr[0])
        self.add_to_mat(self.x1, self.y1, self.x2, self.y2)
        self.convert_position(scmap_arr[2])
        self.add_to_mat(self.x1, self.y1, self.x2, self.y2)
        self.draw_dot()

    def cast_dot(self): # draw line -> click center joint first then other two
        self.centre = self.dot_mat[0]
        self.add_to_mat(self.centre.x1 - self.dot_distance, self.centre.y1, self.centre.x2 - self.dot_distance, self.centre.y2)
        self.add_to_mat(self.centre.x1 + self.dot_distance, self.centre.y1 + self.dot_distance, self.centre.x2 + self.dot_distance, self.centre.y2 + self.dot_distance)
        # self.dot_mat.append(Dot(self.centre.x1-(self.width)*self.cast_dot_scaling_w, self.centre.y1+(self.height)*self.cast_dot_scaling_h, self.centre.x2-(self.width)*self.cast_dot_scaling_w, self.centre.y2+(self.height)*self.cast_dot_scaling_h))
        # self.dot_mat.append(Dot(self.centre.x1+(self.width)*self.cast_dot_scaling_w, self.centre.y1+(self.height)*self.cast_dot_scaling_h, self.centre.x2+(self.width)*self.cast_dot_scaling_w, self.centre.y2+(self.height)*self.cast_dot_scaling_h))

    def get_mouse_position(self, event):
        self.x = event.x
        self.y = event.y
        print(self.x, self.y)

    def within_bounds(self, event):
        x_val = event.x
        y_val = event.y
        TH = 25
        self.x1, self.y1 = (event.x - 1), (event.y - 1)
        self.x2, self.y2 = (event.x + 1), (event.y + 1)
        for i in range(len(self.dot_mat)):
            if (x_val + TH > self.dot_mat[i].x1 and x_val - TH < self.dot_mat[i].x2 and y_val + TH > self.dot_mat[i].y1 and y_val - TH < self.dot_mat[i].y2):
                return i
        return -1

    def drag_dot(self, event):
        index = self.within_bounds(event)
        # print(index)
        if index > -1 and index < 4:
            # print(index)
            self.dot_mat[index] = Dot(self.x1 - 25, self.y1 -25, self.x2 + 25, self.y2 + 25)
            self.draw_dot()
            self.calculate_angle()
            self.place_angle()
        
        # self.x = event.x
        # self.y = event.y
        # print(len(self.dot_mat))
        # for i in range(len(self.dot_mat)): # loop through all to find the closest one
        #     self.ind = i
        #     self.calculate_distance()
        #     self.distance_arr.append(self.distance)
        # print(self.distance_arr)
        # print(self.distance_arr.index(min(self.distance_arr))) # find the index of the smallest distance
        # temp_ind = self.distance_arr.index(min(self.distance_arr))
        # self.x1, self.y1 = (event.x - 1), (event.y - 1)
        # self.x2, self.y2 = (event.x + 1), (event.y + 1)
        # self.dot_mat[temp_ind] = Dot(self.x1, self.x2, self.y1, self.y2)
        # self.ind = temp_ind
        # self.calculate_distance()
        # self.distance_arr[temp_ind] = self.distance
        # # self.x = event.x
        # # self.y = event.y

    def close_overlay(self, event):
        self.root_overlay.iconify()
        self.root_2.destroy()
        self.root_2 = None
        print('minimized window')
        
    def add_dot(self, event):
        # if (event.x - 100 > event.x AND event.x < self.width):
        #     if(event.y > self.height/2 + self.height/4 AND event.y < self.height/2 - self.height/4):
            #     print("Invalid location")
            # else:
        if (len(self.dot_mat) > 2):
            self.dot_mat = []

        self.x1, self.y1 = (event.x), (event.y)
        self.x2, self.y2 = (event.x), (event.y)
        self.add_to_mat(self.x1, self.y1, self.x2, self.y2)
        self.cast_dot()
        self.draw_dot()
        
    def add_to_mat(self, x1, y1, x2, y2):
        self.dot_mat.append((Dot(x1 - 25, y1 -25, x2 + 25, y2 + 25)))

    def calculate_distance(self, d1, d2):
        x1 = d1[0]
        y1 = d1[1]
        x2 = d2[0]
        y2 = d2[1]
        return math.hypot(x2 - x1, y2 - y1)

    def calculate_angle(self):
        d0 = np.array([self.dot_mat[0].x1, self.dot_mat[0].y1])
        d1 = np.array([self.dot_mat[1].x1, self.dot_mat[1].y1])
        d2 = np.array([self.dot_mat[2].x1, self.dot_mat[2].y1])
        d10 = self.calculate_distance(d0, d1)
        
        d12 = self.calculate_distance(d1, d2)
        d02 = self.calculate_distance(d0, d2)
        angle = np.arccos((d10**2 - d12**2 + d02**2)/(2*d10*d02))
        angle = float(np.degrees(angle))
        onepi = np.degrees(np.pi)
        # print(self.movement_ctr)
        # print(self.current_movement)
        if self.current_movement == 'Flexion':
            # print('in here')
            self.angle = round(angle - onepi,2)
        
        else:
            self.angle = round(onepi - angle,2)
        # print(str(self.angle))
        # lol numpy
        # theta = 
        # d

    def make_label(self, root_overlay, x, y, w, h, text, *args, **kwargs):
        f = tk.Frame(root_overlay, width=w, height=h -30)
        f.pack_propagate(0)
        f.place(x=x, y=y)
        label = tk.Label(f, text=text)
        label.pack(side=tk.LEFT, padx=12)
        return label

    def place_angle(self):
        # for i in range(len(self.dot_mat)):
        #     self.canvas.create_oval(self.dot_mat[i].x1, self.dot_mat[i].y1, self.dot_mat[i].x2, self.dot_mat[i].y2, fill=self.array_color[i], outline=self.pin_joint, width=1)
        origin_location = (self.dot_mat[0].center_coordinate[0], self.dot_mat[0].center_coordinate[1])
        d1 = (self.dot_mat[1].center_coordinate[0], self.dot_mat[1].center_coordinate[1])
        d2 = (self.dot_mat[2].center_coordinate[0], self.dot_mat[2].center_coordinate[1])
        location = ((origin_location[0] + d1[0] + d2[0])/3, (origin_location[1] + d1[1] + d2[1])/3)
        
        # self.root_2 = tk.Tk()
        # self.root_2.attributes("-alpha", 1)

        if isinstance(self.root_2, type(None)):
            self.root_2 = tk.Tk()
        
        self.root_2.geometry('%dx%d+%d+%d'%(60, 40, location[0], location[1]))
        self.root_2.overrideredirect(True)
        self.root_2.attributes("-alpha", 0.8)
        self.make_label(self.root_2, 0, 10, 60, 50, str(self.angle) + '\u00B0')

        # w = self.Label(self.root_2, text=str(), font=("Helvetica", 9))
        # w.pack()

    def draw_dot(self):
        # print('drawing DOT')
        self.canvas_overlay.delete("all")
        for i in range(len(self.dot_mat)):
            self.canvas_overlay.create_oval(self.dot_mat[i].x1, self.dot_mat[i].y1, self.dot_mat[i].x2, self.dot_mat[i].y2, fill=self.array_color[i], outline=self.pin_joint, width=1)
        self.calculate_angle()
        self.place_angle()
        self.draw_slopeline()
        
        # self.dot_mat.append(Dot(self.dot_mat.x1, self.dot_mat.y1, self.dot_mat.x2, self.dot_mat.y2, fill="yellow", outline=self.pin_joint, width=5))
        # self.canvas.create_oval(self.dot_mat.x1, self.dot_mat.y1, self.dot_mat.x2, self.dot_mat.y2, fill="yellow", outline=self.pin_joint, width=5)
        # self.cast_dot()

    def extend_line(self, x1, y1, x2, y2, val):
        slope = (y1 - y2) / (x1 - x2)
        if x1 < x2:
            x1_ext = x1 - val
            y1_ext = y1 - val*slope
        else: 
            x1_ext = x1 + val
            y1_ext = y1 + val*slope

        return (x1_ext, y1_ext, x2, y2)


    def draw_slopeline(self):
        centre = self.dot_mat[0]
        first = self.dot_mat[1]
        second = self.dot_mat[2]
        # self.canvas.create_line(0, 0, self.width, self.height, fill="red", width=10)

        self.canvas_overlay.create_line(self.extend_line(centre.center_coordinate[0], centre.center_coordinate[1], first.center_coordinate[0], first.center_coordinate[1], 100), fill="red", width=2)
        self.canvas_overlay.create_line(self.extend_line(centre.center_coordinate[0], centre.center_coordinate[1], second.center_coordinate[0], second.center_coordinate[1], 100), fill="red", width=2)
        self.canvas_overlay.pack()
        if self.loop_SC:
            self.loop_screenshot()

                
    def undo(self, event):
        self.canvas_overlay.delete("all")
        self.root_2.destroy()
        self.root_2 = None

sess, outputs, inputs = prepare_DNN()

if __name__ == '__main__':
    ge = draw_output()

        