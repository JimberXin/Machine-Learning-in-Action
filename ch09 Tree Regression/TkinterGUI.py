# ===============================================================================================
# Author: Junbo Xin
# Date: 2015/02/12
# Description:  Using Tkinter to plot the GUI
# ===============================================================================================

from Tkinter import *
from numpy import *
import regressionTree

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

'''
def re_draw(max_err, min_num):
    pass


def draw_new_tree():
    pass
'''


def re_draw(min_num, max_err):
    re_draw.f.clf()  # clear the figure
    re_draw.a = re_draw.f.add_subplot(111)

    # judge whether the box has been chosen or not
    if check_btn_var.get():
        if min_num < 2:
            min_num = 2
        my_tree = regressionTree.create_tree(re_draw.raw_dat, regressionTree.model_leaf, \
                                             regressionTree.model_err)
        y_hat = regressionTree.create_forecast_tree(my_tree, re_draw.test_dat, regressionTree.model_tree_eval)

    # did not choose the box, use the default choice
    else:
        my_tree = regressionTree.create_tree(re_draw.raw_dat, ops=(min_num, max_err))
        y_hat = regressionTree.create_forecast_tree(my_tree, re_draw.test_dat)

    # use scatter to plot the real value
    re_draw.a.scatter(array(re_draw.raw_dat[:, 0]), array(re_draw.raw_dat[:, 1]), s=5)

    # use plot to plot the estimate value y_hat
    re_draw.a.plot(array(re_draw.test_dat), array(y_hat), linewidth=2.0)
    re_draw.canvas.show()


def get_inputs():
    try:
        min_num = int(min_num_enter.get())
    except:
        min_num = 10
        print 'enter interger for min_num:'
        min_num_enter.delete(0, END)
        min_num_enter.insert(0, '10')
    try:
        max_err = float(max_err_enter.get())
    except:
        max_err = 1.0
        print 'enter Float for max_err:'
        max_err_enter.delete(0, END)
        max_err_enter.insert(0, '1.0')
    return min_num, max_err


def draw_new_tree():
    # get value from entry boxes
    min_num, max_err = get_inputs()
    re_draw(min_num, max_err)


# ===================================== Main Code =========================================
root = Tk()

re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.show()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# my_label = Label(root, text='Plot Place Holder',)
# my_label.grid(row=0, column=1)

# min_num entry
Label(root, text='min num').grid(row=1, column=0)
min_num_enter = Entry(root)
min_num_enter.grid(row=1, column=1)
min_num_enter.insert(0, '10')

# max_err entry
Label(root, text='max err').grid(row=2, column=0)
max_err_enter = Entry(root)
max_err_enter.grid(row=2, column=1)
max_err_enter.insert(0, '1.0')

# redraw button and the check box
Button(root, text='ReDraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)
check_btn_var = IntVar()
check_btn = Checkbutton(root, text='Model Tree', variable=check_btn_var)
check_btn.grid(row=3, column=0, columnspan=2)

# process the data
re_draw.raw_dat = mat(regressionTree.load_data_set('sine.txt'))
re_draw.test_dat = arange(min(re_draw.raw_dat[:, 0]), max(re_draw.raw_dat[:, 0]), 0.01)
re_draw(1.0, 10)

root.mainloop()