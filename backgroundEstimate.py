import cv
import pyglet

class Target:

    def __init__(self): #_init_ is the constructor, self is used like this pointer. Object to the class target. capture is a data member of the class
        self.capture = cv.CaptureFromFile('F:\\Project\\Video1\\actual.mp4') #CaptureFromCAM(0) for webcam videos.
        self.cam = cv.CaptureFromFile('F:\\Project\\Video1\\background.mp4') 
        cv.NamedWindow("Target", 1)
        nFrames = cv.GetCaptureProperty( self.capture , cv.CV_CAP_PROP_FRAME_COUNT );
        fps = cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        #fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        #fps=20.0
        #frame = cv.QueryFrame(self.capture)
        #frame_size = cv.GetSize(frame)
        #self.out=cv.CreateVideoWriter('finalvid.avi',fourcc,fps,frame_size,True)
        #print nFrames
        #print fps
    def run(self):
        # Capture first frame to get size
        countbg=0
        f = cv.QueryFrame(self.cam)
        fps1 = cv.GetCaptureProperty(self.cam, cv.CV_CAP_PROP_FPS)
        print fps1
        first=True
        color_image1 = cv.CreateImage(cv.GetSize(f), 8, 3)
        moving_average1 = cv.CreateImage(cv.GetSize(f), cv.IPL_DEPTH_32F, 3)
        while countbg<fps1:
            color_image1 = cv.QueryFrame(self.cam)
            cv.Smooth(color_image1, color_image1, cv.CV_GAUSSIAN, 3, 0)
            if first:
                cv.ConvertScale(color_image1, moving_average1, 1.0, 0.0) #sort of copying. src,dst,scale,shift-allows type conversion. value added after
                                    #scaling. Optional scaling, shifting.
                temp1 = cv.CloneImage(color_image1)
                #cv.ShowImage("Target",color_image1)
                first = False # for the first frame alone.
            else:           #for all the other frames
                cv.RunningAvg(color_image1, moving_average1, 0.02, None)
                #cv.ShowImage("Target",color_image1)
            cv.ConvertScale(moving_average1, temp1, 1.0, 0.0)
            countbg=countbg+1
            #cv.ShowImage("Target",temp1)
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        #storage1=cv.CreateMemStorage(0)
        #found = list(cv.HOGDetectMultiScale(frame, storage1, win_stride=(8,8),
               # padding=(32,32), scale=1.05, group_threshold=2))
        fps = cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        ##print frame_size
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)# arguments-(width,height), bit depth - no.of bits, no. of channels per pixel
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)

        first = True
        second=True
        n=0
        cp11 = []
        cp22 = []
        center_point1 = []
        count=0
            
        while True:
            #n=n+1
            #print n
            n=n+1
            closest_to_left = cv.GetSize(frame)[0]
            ##print closest_to_left
            closest_to_right = cv.GetSize(frame)[1]
            ##print closest_to_right
            color_image = cv.QueryFrame(self.capture)

            
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)# src, dst, smoothing using gaussian kernal-works with 1,3 channels. can have src,dst
                                #same. 8bit or 32 bit floating point images.

            if first:
                difference = cv.CloneImage(color_image) #fully copies the image.
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0) #sort of copying. src,dst,scale,shift-allows type conversion. value added after
                                    #scaling. Optional scaling, shifting.
                first = False # for the first frame alone.
            else:           #for all the other frames
                cv.RunningAvg(color_image, moving_average, 0.02, None) # second argument must be 32F or 64F.
                            # img, accumulated image-running average of prev frames, alpha - how fast previous images are forgotten, mask-optional.
            #if n>900:
             #second=False
            
            cv.ConvertScale(temp1, temp, 1.0, 0.0) # cannot use clone because of diff in bit depths.
            #cv.ShowImage('Temp',temp)
            
            cv.AbsDiff(color_image, temp, difference) # all three images of same type. src1-src2.

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY) # if range is going to be a problem, scale the image to the required range before using this.

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)#src, dst, thresh, max. value, thresholdtype).
                                #Used only on greyscale images. can be used for removing noise, by changing thresholdtype.

            # Dilate and erode to get people blobs
            cv.Dilate(grey_image, grey_image, None, 18)  # src,dst- should be same type, element=Mat() can be a matrix used for dilation, No. of iterations.
            cv.Erode(grey_image, grey_image, None, 10) #replaces pixel with min value of all the pixels in the neighbourhood specified by the element
                                #matrix. Dilation replaces the pixel with max value.
            #cv.ShowImage("GreyScale", grey_image)
            storage = cv.CreateMemStorage(0) #0 allocates default value of 64K memory bytes.
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE) # performs only on binary images.src= 8bit 1channel,
                    #countours- vectors of pts. src changed while finding countours. 3rd arg mode- gets countours and puts them in a 2 level hirarchy.
                    # 4th arg method- gets only the endpoints. if rectangular- 4 pts are extracted.

            points = []
            
            i=0
            k=0
            while contour:# if contours are found.
                area=cv.ContourArea(list(contour))
                if (area > 12000.0):
                #print area
                    bound_rect = cv.BoundingRect(list(contour)) # calculates the up-right bounding rect for the given point set.
                #contour = contour.h_next() #to iterate through all countours.
                    pt1 = (bound_rect[0], bound_rect[1])
                    pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                    points.append(pt1)
                    points.append(pt2)
                
                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 1)
                
                    cp1 = bound_rect[0] + (bound_rect[2]/2)
                    cp2 = bound_rect[1] + (bound_rect[3]/2)
                    cp11.append(cp1)
                    cp22.append(cp2)
                    
                #print cp1, cp2
                    #if len(points):
                    #center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
                    k=k+1
                
                    #center_point1.append(center_point)
                    count=count+1
                contour = contour.h_next()
                #print center_point,cp1,cp2
                    #cv.Circle(color_image, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
                    #cv.Circle(color_image, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
                    #cv.Circle(color_image, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
            ##print count,cp11,cp22,n
            while(i<count):
                #if (cp11[i]-cp11[i-1] < 1 and cp22[i]-cp22[i-1] < 1):
                 #   cv.Circle(color_image, (cp11[i-1], cp22[i-1]), 1, cv.CV_RGB(255, 100, 0), 1)
                #else:
                cv.Circle(color_image, (cp11[i], cp22[i]), 1, cv.CV_RGB(255, 100, 0), 1)
                    #if i>0 :
                        #cv.Line(color_image, center_point1[i], center_point1[i-1], cv.CV_RGB(255, 100, 0) , 1, 8)  
            #if(n==5):
               # print cp11[2],cp22[2]
                    #cv.Circle(color_image, center_point1[i], 1, cv.CV_RGB(255, 100, 0), 1)
                i=i+1
            print i
            cv.ShowImage("Target", color_image)
           
        #Listen for ESC key
            c = cv.WaitKey(int(fps))  # to have a wait time between the displayed frames. Should be there after every showimage call
            if c == 27:  #  in case escape was pressed to close window. Hexcoded as 27, I think.
                break


if __name__=="__main__":
    t = Target()
    t.run()

    
