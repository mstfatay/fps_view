# fps_view
FpsView is a class to draw useful information of a drone on its fps camera view.

Try this:
>>>import fpsview
# img is a numpy.ndarray representing image
>>>fv = fpsview.fpsview(img.shape)   #use that line only once 
>>>fv.fpsView(img, roll=0, hdg=1, throttle=50, speed=35,...)   #call it for every frame
