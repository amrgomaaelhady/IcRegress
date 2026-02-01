"""
Environment for Robot Arm.
You can customize this script in a way you want.
View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet
import time
import pandas as pd
import torch
import random
from pyglet.image import load
from sklearn.metrics import mean_squared_error
import os
import glob

# Fetch the only available CSV file in the current directory
current_folder = os.getcwd()
csv_files = glob.glob(os.path.join(current_folder, "*.csv"))

# Check if there's only one CSV file
if len(csv_files) != 1:
    raise ValueError("There should be exactly one CSV file in the current directory.")

# Use the only CSV file found
FILE = csv_files[0]
filename = os.path.basename(FILE)

# pyglet.clock.run(10000)




class ArmEnv(object):
    action_bound = [-1, 1]
    action_dim = 1
    state_dim =3
    dt = .01  # refresh rate
    viewer = None
    viewer_xy = (800, 800)
    get_point = False
    grab_counter = 0


    def __init__(self, mode='easy'):
        self.mode = mode
        self.center_coord = np.array(self.viewer_xy) / 2
        self.df1 = pd.read_csv(filename)
        #print(self.df1.columns)
        #self.df2 = pd.read_csv("predictions.csv")
        #self.df1["normalized_p_a"] = self.df1["angle_p"]
        #self.df1["normalized_g_a"] = self.df1["angle_g"]
        #self.df1["normalized_difference"] = self.df1["difference"]
        #self.df1["normalized_difference"] = np.clip(self.df1["normalized_difference"], -1, 1)
        self.pointing = 0 #self.df1["horizontal_angle_finger_pointing_unscaled"][0]
        self.target = self.df1["gtv_average"][0]
        self.gazing= 0 # self.df1["eyegaze_horiz_angle_x"][0]
        self.px =0
        self.pz =0
        self.gx = 0
        self.gz = 0
        self.hx = 0
        self.hz = 0
        self.hgx=0
        self.hgz=0
        self.modality_difference = 0
        self.side = 0
        self.lg = 0
        self.lp = 0
        self.p_a = 0
        self.g_a =0
        self.reg = 0
        self.index =0
        self.prediction= 0
        self.temp1 =1000
        self.count = 0
        #print(self.point_info)

    def step(self, action, index):
        # action bound between -1 to 1 and the angle for each bar should be around -2Pi to 2Pi
        # here it try to map the action bound with the angle boun
        self.index = index
        self.pointing = 0# self.df1["horizontal_angle_finger_pointing_unscaled"][index]
        self.target = self.df1["gtv_average"][index]
        self.gazing = 0# self.df1["eyegaze_horiz_angle_x"][index]

        prediction = action[0]
        s , l = self._get_state(index)
        self.prediction= prediction
        #r =
        #r = - np.sqrt( ( prediction - self.target) ** 2 )
        #print(r)
        #r = - abs(prediction - self.target)
        #print(index)
        #print(self.df1['gtv_min_average'][index])
        #print(self.df1['gtv_max_average'][index])
        #print(self.df1['gtv_average'][index])
        hbw = abs(self.df1['gtv_min_average'][index] - self.df1['gtv_max_average'][index])/2
        #print(hbw)
        #r = - abs( prediction - self.target) / hbw
        if(prediction >=self.df1['gtv_min_average'][index] and prediction <= self.df1['gtv_max_average'][index]):
            r=25
            #self.get_point = True

        else:
            r= - abs( prediction - self.target) / hbw - abs( prediction - self.target)
        #print(r)
        #if ( abs( prediction - self.target) / hbw  < 1):
         #   self.get_point = True
         #   r = 25
        #print(r)
        #print(r)
        '''if (abs( prediction - self.target)) < 1:
            self.count += 1
            if (self.count > 2):
                self.get_point = True
                r = 25
        else:
        #    r = -1
            self.count = 0'''
        '''if temp < self.temp1:
            r= 1
            self.temp1= temp
            self.count += 1
            #print(self.count)
            if (self.count > 4):
                self.get_point = True
                r=2
        else:
            r = -0.5
            self.count = 0
            self.temp1 = temp'''
        #if abs( prediction - self.target) > 10:
            #temp = abs((prediction + action[1] * 15) - self.target)
            #self.prediction =prediction + action[1] * 10
        #else:
          #  temp = abs((prediction + action[1] * 9) - self.target)
           # self.prediction = prediction + action[1] * 4


        #print((self.prediction+ action[1] * 15), self.target )

        #print(temp)
        '''if (temp < 1 ):
            self.count += 1
            #print(temp)
            r = 2
            if(self.count > 4):
                self.get_point= True
        elif (temp < self.temp1) :
            r = 1
            self.count = 0
        else :
            r= -1
            self.count = 0

        self.temp1 = temp'''
        #print(self.gazing)
        return s, r, self.get_point

    def reset(self, ind):
        self.get_point = False
        self.grab_counter = 0
        self.count = 0
        self.temp1= 1000
        #ind = random.randint(1, 320)
        #self.modality_difference = self.df1["normalized_difference"][ind]
        #self.reg = self.df1["r_l"][ind]
        #self.side = self.df1["side"][ind]
        self.px = self.df1["pta_average"][ind]
        #self.pz = self.df1["ptz_1s_avg"][ind]
        self.gx = self.df1["gza_average"][ind]
        #self.gz = self.df1["gzz_1s_avg"][ind]
        self.hx = self.df1["hpa_average"][ind]
        #self.hz = self.df1["hpz_1s_avg"][ind]
        self.hgx = self.df1["hga_average"][ind]
        #self.hgz = self.df1["hpx_1s_avg"][ind]
        #self.x = self.df1["x"][ind]
        #self.y = self.df1["y"][ind]
        #self.z = self.df1["z"][ind]
        #self.gx = self.df1["gx"][ind]
        #self.gy = self.df1["gy"][ind]
        #self.p_a = self.df1["normalized_p_a"][ind]
        #self.g_a = self.df1["normalized_g_a"][ind]
        #self.modality_difference = self.df1["normalized_difference"][ind]
        #self.p_a = self.df1["normalized_p_a"][ind]
        #self.g_a = self.df1["normalized_g_a"][ind]
        self.pointing= 0 # self.df1["horizontal_angle_finger_pointing_unscaled"][ind]
        #print(ind)
        self.target = self.df1["gtv_average"][ind]
        self.gazing = 0 #self.df1["eyegaze_horiz_angle_x"][ind]
        return self._get_state(ind)[0]

    def render(self, x):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.pointing, self.target, self.gazing, self.prediction, self.df1)
        self.viewer.render(x, self.prediction)

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def _get_state(self,ind):
        # return the distance (dx, dy) between arm finger point with blue point
        '''self.modality_difference = (self.df1["difference"][ind])
        self.side = self.df1["side"][ind]
        self.x = self.df1["x"][ind]
        self.y = self.df1["y"][ind]
        self.z = self.df1["z"][ind]
        self.gx = self.df1["gx"][ind]
        self.gy = self.df1["gy"][ind]
        self.reg = self.df1["r_l"][ind]
        self.modality_difference = self.df1["normalized_difference"][ind]
        self.p_a = self.df1["normalized_p_a"][ind]
        self.g_a = self.df1["normalized_g_a"][ind]'''
        '''self.px = self.df1["ptx_1s_avg"][ind]
        self.pz = self.df1["ptz_1s_avg"][ind]
        self.gx = self.df1["gzx_1s_avg"][ind]
        self.gz = self.df1["gzz_1s_avg"][ind]
        self.hx = self.df1["hpx_1s_avg"][ind]
        self.hz = self.df1["hpz_1s_avg"][ind]
        self.hgx = self.df1["hgx_1s_avg"][ind]
        self.hgz = self.df1["hpx_1s_avg"][ind]'''
        self.px = self.df1["pta_average"][ind]
        #self.pz = self.df1["ptz_1s_avg"][ind]
        self.gx = self.df1["gza_average"][ind]
        #self.gz = self.df1["gzz_1s_avg"][ind]
        self.hx = self.df1["hpa_average"][ind]
        #self.hz = self.df1["hpz_1s_avg"][ind]
        self.hgx = self.df1["hga_average"][ind]
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([self.gx, self.px, self.hgx]), ind


class Viewer(pyglet.window.Window):
    color = {'background': [1] * 3 + [1]}

    def __init__(self, width, height, pointing, target,gazing, prediction,data):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm',
                                     vsync=False)
        self.set_location(x=0, y=0)
        pyglet.gl.glClearColor(*self.color['background'])
        self.batch = pyglet.graphics.Batch()
        self.center_coord = np.array((min(width, height) / 2,) * 2)
        self.car_image = load('car.png')  # Replace with the actual filename of your car image
        self.car_sprite = pyglet.sprite.Sprite(
            self.car_image,
            x=width // 2 + self.car_image.width // 4,  # Center x-coordinate
            y=height // 2 + self.car_image.height // 4  # Center y-coordinate
        )

        ####################

        self.car_sprite.scale = 0.5
        num_lines = int(360 / 15)  # Calculate the number of lines
        angle_step = 15  # Step size in degrees
        # Draw lines every 15 degrees
        # Calculate the endpoints of each line segment
        lines = []
        for i in range(num_lines):
            angle_rad = np.radians(i * angle_step)
            x1 = 400 + 100 * np.cos(angle_rad)  # Adjust the length as needed
            y1 = 400 + 100 * np.sin(angle_rad)  # Adjust the length as needed
            x2 = 400 + 120 * np.cos(angle_rad)  # Adjust the length as needed
            y2 = 400 + 120 * np.sin(angle_rad)  # Adjust the length as needed
            lines.extend((int(x1), int(y1), int(x2), int(y2)))
        self.lines = self.batch.add(num_lines * 2, pyglet.gl.GL_LINES, None, ('v2i', lines),
                                        ('c3B', (0, 0, 255) * num_lines * 2))

            # Draw X and Y axes in red
        self.axes = self.batch.add(4, pyglet.gl.GL_LINES, None,
                                       ('v2i', (0, 400, 800, 400, 400, 0, 400, 800)),
                                       ('c3B', (255, 0, 0) * 4)
                                       )



        ###################



        self.data = data
        #self.pred_data = pred_data
        self.pointing = pointing
        self.target = target
        self.gazing = gazing
        self.index =0
        self.box_x = 0
        self.box_y = 0
        self.box_angle = 0
        box_size =5
        self.prediction = prediction

        self.angle_label = pyglet.text.Label('', font_name='Arial', font_size=12,
                                             x=self.box_x, y=self.box_y + 30,
                                             anchor_x='center', anchor_y='center',
                                             color=(0, 0, 0, 255),
                                             batch=self.batch)
        self.pointing_label = pyglet.text.Label('', font_name='Arial', font_size=12,
                                                x=self.box_x, y=self.box_y + 50,
                                                anchor_x='center', anchor_y='center',
                                                color=(0, 0, 0, 255),
                                                batch=self.batch)
        self.gazing_label = pyglet.text.Label('', font_name='Arial', font_size=12,
                                                x=self.box_x, y=self.box_y + 50,
                                                anchor_x='center', anchor_y='center',
                                                color=(0, 0, 0, 255),
                                                batch=self.batch)
        self.prediction_label = pyglet.text.Label('', font_name='Arial', font_size=12,
                                              x=self.box_x, y=self.box_y + 50,
                                              anchor_x='center', anchor_y='center',
                                              color=(0, 0, 0, 255),
                                              batch=self.batch)
        self.box1= self.batch.add(4, pyglet.gl.GL_QUADS, None,
                       ('v2i', (int(self.box_x - box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y + box_size),
                                int(self.box_x - box_size), int(self.box_y + box_size))),
                       ('c3B', (0, 255, 0) * 4)  # Green color
                       )
        self.box2=self.batch.add(4, pyglet.gl.GL_QUADS, None,
                       ('v2i', (int(self.box_x - box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y + box_size),
                                int(self.box_x - box_size), int(self.box_y + box_size))),
                       ('c3B', (255, 0, 0) * 4)  # Red color
                       )
        self.box3 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2i', (int(self.box_x - box_size), int(self.box_y - box_size),
                                            int(self.box_x + box_size), int(self.box_y - box_size),
                                            int(self.box_x + box_size), int(self.box_y + box_size),
                                            int(self.box_x - box_size), int(self.box_y + box_size))),
                                   ('c3B', (0, 0, 255) * 4)  # Blue color
                                   )
        self.box4 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2i', (int(self.box_x - box_size), int(self.box_y - box_size),
                                            int(self.box_x + box_size), int(self.box_y - box_size),
                                            int(self.box_x + box_size), int(self.box_y + box_size),
                                            int(self.box_x - box_size), int(self.box_y + box_size))),
                                   ('c3B', (255, 0, 255) * 4)  # Blue color
                                   )
    def render(self, x, prediction):

        self.pointing = 0 #self.data["horizontal_angle_finger_pointing_unscaled"][x]
        self.target = self.data["gtv_average"][x]
        self.gazing = 0 #self.data["eyegaze_horiz_angle_x"][x]
        self.prediction = prediction
       # print(self.prediction)
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def _update_arm(self):

        box_size = 5
       # self.ax1.vertices = (0, 400, 800, 400)
        angle_rad = np.radians(self.target)
        self.box_x = 400 + 150 * np.cos(angle_rad)
        self.box_y = 400 + 150 * np.sin(angle_rad)
        self.box_angle = self.target
        self.angle_label.x = self.box_x
        self.angle_label.y = self.box_y +2
        self.angle_label.text = f'{self.target:.1f}'
        self.box1.vertices= (int(self.box_x - box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y - box_size),
                                int(self.box_x + box_size), int(self.box_y + box_size),
                                int(self.box_x - box_size), int(self.box_y + box_size))
        #print(self.box1.vertices[0], self.box1.vertices[1], self.box1.vertices[2])

        angle_rad = np.radians(self.pointing)
        self.box_x = 400 + 150 * np.cos(angle_rad)
        self.box_y = 400 + 150 * np.sin(angle_rad)
       # self.box2.vertices = (int(self.box_x - box_size), int(self.box_y - box_size),
        #                        int(self.box_x + box_size), int(self.box_y - box_size),
         #                       int(self.box_x + box_size), int(self.box_y + box_size),
          #                      int(self.box_x - box_size), int(self.box_y + box_size))
        #self.pointing_label.x = self.box_x
        #self.pointing_label.y = self.box_y + 2
        #self.pointing_label.text = f'{self.pointing:.1f}'

        angle_rad = np.radians(self.gazing)
        self.box_x = 400 + 150 * np.cos(angle_rad)
        self.box_y = 400 + 150 * np.sin(angle_rad)
        #self.box3.vertices = (int(self.box_x - box_size), int(self.box_y - box_size),
         #                     int(self.box_x + box_size), int(self.box_y - box_size),
          #                    int(self.box_x + box_size), int(self.box_y + box_size),
           #                   int(self.box_x - box_size), int(self.box_y + box_size))
        #self.gazing_label.x = self.box_x
        #self.gazing_label.y = self.box_y + 2
        #self.gazing_label.text = f'{self.gazing:.1f}'


        angle_rad = np.radians(self.prediction)
        self.box_x = 400 + 150 * np.cos(angle_rad)
        self.box_y = 400 + 150 * np.sin(angle_rad)
        self.box4.vertices = (int(self.box_x - box_size), int(self.box_y - box_size),
                              int(self.box_x + box_size), int(self.box_y - box_size),
                              int(self.box_x + box_size), int(self.box_y + box_size),
                              int(self.box_x - box_size), int(self.box_y + box_size))
        self.prediction_label.x = self.box_x
        self.prediction_label.y = self.box_y + 2
        self.prediction_label.text = f'{self.prediction:.1f}'


    def on_draw(self):
        self.clear()
        self.car_sprite.rotation = 180  # Rotate the car sprite to face positive x-axis
        self.car_sprite.draw()
        self.batch.draw()
        self.label.draw()


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        random_int = random.randint(0, 2166)
        env.render(25)
        env.step(env.sample_action(),1670)
        time.sleep(1)
