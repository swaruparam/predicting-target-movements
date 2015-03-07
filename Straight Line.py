import cv

class Target:

    def __init__(self): #_init_ is the constructor, self is used like this pointer. Object to the class target. capture is a data member of the class
         #CaptureFromCAM(0) for webcam videos.
        self.capture = cv.CaptureFromFile('F:\\Project\\Video9\\straight.m2ts') 
        cv.NamedWindow("Target", cv.CV_WINDOW_NORMAL)

       # cv.ResizeWindow("Target",600,600)
        #fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        #fps=20.0
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        print frame_size
        #self.out=cv.CreateVideoWriter('finalvid.avi',fourcc,fps,frame_size,True)

    def run(self):
        
            
        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        #frame = cv.CreateMat(fr.rows/6, fr.cols/6, fr.type)
        #cv.Resize(im, frame)
        #cv.ShowImage("target",frame)
        frame_size = cv.GetSize(frame)
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        writer=cv.CreateVideoWriter('output.avi',fourcc,fps,cv.GetSize(frame),1)
    

            # 
        ##print frame_size
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)# arguments-(width,height), bit depth - no.of bits, no. of channels per pixel
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)
        # Create Kalman Filter
        kalman = cv.CreateKalman(4, 2, 0)
        kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        velocity=1
        track_start=0
        first = True
        second=True
        n=0
        cp11 = []
        cp22 = []
        center_point1 = []
        predict_pt1 = []
        xmove = []
        count=0
        kalmancount=0
        enterloop=0
        n1=-1    
        while True:
            #if n==612:
             #   cv.SaveImage("11.jpg",color_image)
                
            n=n+1
            #print n
            closest_to_left = cv.GetSize(frame)[0]
            ##print closest_to_left
            closest_to_right = cv.GetSize(frame)[1]
            ##print closest_to_right
            color_image = cv.QueryFrame(self.capture)
            #fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
            #fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
            #writer=cv.CreateVideoWriter('output.avi',fourcc,fps,cv.GetSize(frame),1)
    

            # Smooth to get rid of false positives
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
            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1.0, 0.0) # cannot use clone because of diff in bit depths.

            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, difference) # all three images of same type. src1-src2.

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY) # if range is going to be a problem, scale the image to the required range before using this.

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)#src, dst, thresh, max. value, thresholdtype).
                                #Used only on greyscale images. can be used for removing noise, by changing thresholdtype.
            if n==n1+1:
                cv.SaveImage("11.jpg",grey_image)
            # Dilate and erode to get people blobs
            cv.Dilate(grey_image, grey_image, None, 18)  # src,dst- should be same type, element=Mat() can be a matrix used for dilation, No. of iterations.
            if n==n1+1:
                cv.SaveImage("22.jpg",grey_image)
            cv.Erode(grey_image, grey_image, None, 10) #replaces pixel with min value of all the pixels in the neighbourhood specified by the element
                                #matrix. Dilation replaces the pixel with max value.
            if n==n1+1:
                cv.SaveImage("33.jpg",grey_image)
            cv.ShowImage("GreyScale", grey_image)
            storage = cv.CreateMemStorage(0) #0 allocates default value of 64K memory bytes.
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE) # performs only on binary images.src= 8bit 1channel,
                    #countours- vectors of pts. src changed while finding countours. 3rd arg mode- gets countours and puts them in a 2 level hirarchy.
                    # 4th arg method- gets only the endpoints. if rectangular- 4 pts are extracted.

            points = []
            i=0
            k=0
            total=0

            while contour:
                # if contours are found.
                area=cv.ContourArea(list(contour))
                #print area
                if (area > 20000.0):
                    if n1==-1:
                        n1=615
                    track_start=1
                    enterloop=1
                    bound_rect = cv.BoundingRect(list(contour)) # calculates the up-right bounding rect for the given point set.
                    pt1 = (bound_rect[0], bound_rect[1])
                    pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                    points.append(pt1)
                    points.append(pt2)
                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                    cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                    
                    cp1 = bound_rect[0] + (bound_rect[2]/2)
                    cp2 = bound_rect[1] + (bound_rect[3]/2)
                    cp11.append(cp1)
                    cp22.append(cp2)
                    #print bound_rect[3]
                #print cp1, cp2
                    #if len(points):
                    #center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
                   # print "Tracking",cp1,cp2
                    if count>0:
                        if cp11[count]>cp11[count-3]:
                            velocity=cp11[count]-cp11[count-3]
                        else:
                            velocity=cp11[count-3]-cp11[count]
                    
                    #if velocity==0:
                     #   velocity=5
                    #print velocity
                    #center_point1.append(center_point)
                    #if count>0:
                     #   xdash=cp11[count]-cp11[count-3]

                      #  if xdash<0:
                       #     xdash=-cp11[count-3]+cp11[count]
                        #xmove.append(xdash)
                        #while(k<count):
                            #print "Stuck"
                         #   total=total+xmove[k]
                          #  k=k+1
                        #if total==0:
                         #   velocity=6
                        #else:
                         #   velocity=total/count
                        #print velocity
                    count=count+1
                    #kalmancount=kalmancount+1
                    #print count,kalmancount
                    
                    
                contour = contour.h_next() #to iterate through all countours.
            if track_start==1 and enterloop==0:
                    #print "Occlusion"
                    print "Occlsion",velocity
             
                    cp1=int(kalman_prediction[0,0])+velocity
                    cp2=int(kalman_prediction[1,0])
                    #print cp1,cp2
            if track_start==1:
                # set previous state for prediction
                kalman.state_pre[0,0]  = cp1
                kalman.state_pre[1,0]  = cp2
                kalman.state_pre[2,0]  = 0
                kalman.state_pre[3,0]  = 0
            #print kalman.state_pre[0,0]

            # set kalman transition matrix
                kalman.transition_matrix[0,0] = 1
                kalman.transition_matrix[0,1] = 0
                kalman.transition_matrix[0,2] = 0
                kalman.transition_matrix[0,3] = 0
                kalman.transition_matrix[1,0] = 0
                kalman.transition_matrix[1,1] = 1
                kalman.transition_matrix[1,2] = 0
                kalman.transition_matrix[1,3] = 0
                kalman.transition_matrix[2,0] = 0
                kalman.transition_matrix[2,1] = 0
                kalman.transition_matrix[2,2] = 0
                kalman.transition_matrix[2,3] = 1
                kalman.transition_matrix[3,0] = 0
                kalman.transition_matrix[3,1] = 0
                kalman.transition_matrix[3,2] = 0
                kalman.transition_matrix[3,3] = 1
 
            # set Kalman Filter
                cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
                cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))
                cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
                cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(1))
 
            #Prediction
                kalman_prediction = cv.KalmanPredict(kalman)
                predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
                predict_pt1.append(predict_pt)
                #print "Prediction",predict_pt
            #Correction
                kalman_estimated = cv.KalmanCorrect(kalman, kalman_measurement)
                state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])

            #measurement
                kalman_measurement[0, 0] = cp1
                kalman_measurement[1, 0] = cp2
                kalmancount=kalmancount+1
            
            while(i<kalmancount):
                pp1=predict_pt1[i]
                cv.Circle(color_image, (pp1[0]+10,pp1[1]+10), 1, cv.CV_RGB(0, 255, 0), 3)
                i=i+1
            i=0

            while(i<count):
                   cv.Circle(color_image, (cp11[i], cp22[i]), 1, cv.CV_RGB(255, 100, 0), 3)
                   i=i+1 
            #if(n==5):
               # print cp11[2],cp22[2]
                    #cv.Circle(color_image, center_point1[i], 1, cv.CV_RGB(255, 0, 0), 3)
                
            cv.ShowImage("Target", color_image)
            if enterloop==1:
                enterloop=0
            cv.WriteFrame(writer,color_image)
            # Listen for ESC key
            c = cv.WaitKey(int(fps/20))  # to have a wait time between the displayed frames. Should be there after every showimage call
            if c == 27:  #  in case escape was pressed to close window. Hexcoded as 27, I think.
                break

if __name__=="__main__":
    t = Target()
    t.run()
