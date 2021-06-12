import _thread
import time
import tkinter as tk
from tkinter import Tk, Label
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import multiprocessing
from multiprocessing import Process
import os
from process import *

window_width = 856
window_height = 480
image_width = int(window_width * 1.0)
image_height = int(window_height * 1.0)
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450
video_dir = 'data_for_test/'
buffer_frame_list = []
buffer_frame_already_shown_indicator_list = []

def show_video(buffer_frame_list, buffer_frame_already_shown_indicator_list):
    while(len(buffer_frame_list) > 0):
        cv2.imshow(buffer_frame_list[0])
        buffer_frame_list.pop(0)
        buffer_frame_already_shown_indicator_list.pop(0)

def start_tracking():
    global video_dir
    val = theLB.get()  

    opt = {}
    opt['cfg'] = 'pose_module/experiments/coco/pose.yaml'
    opt['modelDir'] = ''
    opt['logDir'] = ''
    opt['weights'] = ['weights/det.pt'] 
    opt['source'] = os.path.join(video_dir, val) 
    opt['ab_detect_hyp'] = ''
    opt['output'] = 'cache/' #
    opt['track_out'] = '' + val 
    opt['img_size'] = 1920
    opt['conf_thres'] = 0.25
    opt['iou_thres'] = 0.5
    opt['device'] = '1'
    opt['view_img'] = False
    opt['save_txt'] = False
    opt['agnostic_nms'] = False
    opt['augment'] = False
    opt['update'] = False
    opt['classes'] = None
    opt['similarity_module_config_file'] = 'similarity_module/configs/im_osnet_x0_25_softmax_256x128_amsgrad.yaml'
    opt['show_anomaly_or_foreignmatter_or_tracking'] = 0
    opt['opts'] = ['model.load_similarity_weights',
                   'similarity_module/model/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', \
                   'test.evaluate_similarity_module', 'True']
    detect(opt, canvas1, win, buffer_frame_list)
    # video(video_dir+val)


def tkImage(frame):
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage

def video(video_dir, canvas1, win):
    def video_loop(video_dir):
        img_list = os.listdir(video_dir)
        img_list.sort()
        for img_name in img_list:
            frame = cv2.imread(os.path.join(video_dir, img_name))
            picture1 = tkImage(frame)
            canvas1.create_image(0, 0, anchor='nw', image=picture1)
            win.update_idletasks()
            win.update()

    video_loop(video_dir)

win = tk.Tk()
win.title("Tracking")
win.geometry(str(window_width+150) + 'x' + str(window_height))
# video menu list
# place menu list on the window
mlist = os.listdir(video_dir)
# ttk.Label(win, text = "GFG Combobox Widget", background = 'green', foreground ="white", font = ("Times New Roman", 15)).grid(row = 0, column = 1)
ttk.Label(win, text = "Select video :", font = ("Times New Roman", 10)).place(relx=0, x=875, y=70, anchor="nw") # .grid(column = 1, row = 0, sticky='W') #.grid(column = 0, row = 5, padx = 10, pady = 25)
n = tk.StringVar()
theLB = ttk.Combobox(win, width=12, height=2, textvariable = n)
theLB['values'] = mlist
theLB.grid(column = 15, row = 1, sticky = 'NW')
theLB.current(0)  #
theLB.place(x=880, y=90)
# place image on the window
canvas1 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas1.place(x=imagepos_x, y=imagepos_y)
# canvas1.bind('<Button-1>', left)
# canvas1.bind('<Button-3>', right)
# place button on the window
B2 = Button(win, text="Tracking Visualization", command=start_tracking, fg='red', width=15, height=2)
B2.place(x=865, y=20)


if __name__ == '__main__':
    # video(video_dir)
    win.mainloop()
#     Process(target=show_video).start()
# #     cv2.destroyAllWindows()


