import cv

class Target:

    def __init__(self): #_init_ is the constructor, self is used like this pointer. Object to the class target. capture is a data member of the class
        self.capture = cv.CaptureFromFile('F:\\Project\\Video9\\sep3.m2ts') #CaptureFromCAM(0) for webcam videos.
        #self.capture = cv.CaptureFromFile('F:\\Project\\Video5\\v3.m2ts')
        #self.capture = cv.CaptureFromFile('F:\\Project\\Video1\\actual.mp4')
        #self.capture = cv.CaptureFromFile('C:\\Users\\2124718\\Videos\\Movavi Library\\v1.avi')
        cv.NamedWindow("Target",cv.CV_WINDOW_NORMAL)
        cv.NamedWindow("GreyScale",cv.CV_WINDOW_NORMAL)
        #fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        #fps=20.0
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        print frame_size
        #self.out=cv.CreateVideoWriter('finalvid.avi',fourcc,fps,frame_size,True)

    def run(self):
        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        writer=cv.CreateVideoWriter('outputtry2obj.avi',fourcc,fps,cv.GetSize(frame),1)
        #width = 400 #leave None for auto-detection
        #height = 400
        #cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height)
        #cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height)
        #frame = cv.QueryFrame(self.capture)
        #frame_size = cv.GetSize(frame)
        #print frame_size
        #frame = cv.CreateMat(fr.rows/6, fr.cols/6, fr.type)
        #cv.Resize(im, frame)
        #cv.ShowImage("target",frame)
        frame_size = cv.GetSize(frame)
        w=frame_size[0]
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        ##print frame_size
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)# arguments-(width,height), bit depth - no.of bits, no. of channels per pixel
        
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)
        # Create Kalman Filter for contour1
        kalman = cv.CreateKalman(4, 2, 0)
        kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        #for contour2
        kalman1 = cv.CreateKalman(4, 2, 0)
        kalman_state1 = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_process_noise1 = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_measurement1 = cv.CreateMat(2, 1, cv.CV_32FC1)

        velocity=1
        track_start=0
        first = True
        second=True
        n=0
        cp11 = []
        cpnew11=[]
        cpnew22=[]
        cp22 = []
        center_point1 = []
        predict_pt1 = []
        predict_pt2=[]
        cmerge1=[]
        cmerge2=[]
        a=b=c=0
        xmove = []
        count=0
        newcount=0
        mergecount=0
        kalmancount=0
        kalmancount1=0
        enterloop=0
        notwo=0
        noc=0
        swap=0
        two=0
        n8=-1
        #singleobj=[]    
        while True:
            if n==n8:
                cv.SaveImage("11.jpg",color_image)
                print "check"
            
            n=n+1
            closest_to_left = cv.GetSize(frame)[0]
            ##print closest_to_left
            closest_to_right = cv.GetSize(frame)[1]
            ##print closest_to_right
            color_image = cv.QueryFrame(self.capture)

            # Smooth to get rid of false positives
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)# src, dst, smoothing using gaussian kernal-works with 1,3 channels. can have src,dst
                                #same. 8bit or 32 bit floating point images.

            if first:
                difference = cv.CloneImage(color_image) #fully copies the image.
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0) #sort of copying. src,dst,scale,shift-allows type conversion. value added after
                                    #scaling. Optional scaling, shifting.
                first = False # for the first frame alone.
            elif(n<=150):           #for all the other frames
               cv.RunningAvg(color_image, moving_average, 0.02, None) # second argument must be 32F or 64F.
                            # img, accumulated image-running average of prev frames, alpha - how fast previous images are forgotten, mask-optional.
            #if n==150:
             #   cv.ShowImage("Target", color_image)
             #second=False
            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1.0, 0.0) # cannot use clone because of diff in bit depths.
            cv.SaveImage("33.jpg",temp)

            # Minus the current frame from the moving average.
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
            cv.ShowImage("GreyScale", grey_image)
            if n==n8+1:
                print "check"
                
                cv.SaveImage("22.jpg",grey_image)
            #cv.WriteFrame(writer,grey_image)
            storage = cv.CreateMemStorage(0) #0 allocates default value of 64K memory bytes.
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_NONE) # performs only on binary images.src= 8bit 1channel,
                    #countours- vectors of pts. src changed while finding countours. 3rd arg mode- gets countours and puts them in a 2 level hirarchy.
                    # 4th arg method- gets only the endpoints. if rectangular- 4 pts are extracted.
            contour1 = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_NONE)
            if noc==1:
                check=1
            else:
                check=0
            noc=0
            while contour1:
                area=cv.ContourArea(list(contour1))
                if area>67000.0:
                    noc=noc+1
                contour1=contour1.h_next()
                
                
            points = []
            i=0
            kk=0
            total=0
            #if (track_start==0):
             #   cp11[1]=0

            while contour:# if contours are found.
                #print n
                filled_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
                area=cv.ContourArea(list(contour))
                if (area > 67000.0):
                   # print area
                    if noc==1:   ####Only one contour
                        #print "NOC:",noc
                        #if n1==-1:
                         #   n1=n
                        track_start=1
                        enterloop=1
                        bound_rect=cv.BoundingRect(list(contour))
                        pt1 = (bound_rect[0], bound_rect[1])
                        pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                        points.append(pt1)
                        points.append(pt2)
                        cp1 = bound_rect[0] + (bound_rect[2]/2)
                        cp2 = bound_rect[1] + (bound_rect[3]/2)
                        if count==0 or two==0:
                            #if n2==-1:
                             #   n2=n
                           # print "check"
                            a=1
                            cp11.append(cp1)
                            cp22.append(cp2)
                            count=count+1
                            cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                            font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                            cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                        
                            
                        else:
                            
                            #print "check"
                            diff= cp11[count-1]-cpnew11[newcount-1]
                            if diff < 0:  ### incase of neg make it pos
                                diff=0-diff
                            if diff> int (w/4.0): ###in this case, there is only one person in the frame. Other has gone out the other side
                                #print "check"
                                ck=cp1-cp11[count-1]
                                cj=cp1-cpnew11[newcount-1]
                                if ck<0:
                                    ck=0-ck
                                if cj<0:
                                    cj=0-cj
                                if ck-cj>0: ### find which diff is closest to zero and update the corresponding center
                                    #if n3==-1:
                                     #   n3=n
                                    cpnew11.append(cp1)
                                    cpnew22.append(cp2)
                                    newcount=newcount+1
                                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                    cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                                    b=1
                                else:
                                    #if n4==-1:
                                     #   n4=n
                                    cp11.append(cp1)
                                    cp22.append(cp2)
                                    count=count+1
                                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                    cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                                    a=1
                            else: ### in this case two people have merged
                     #           print "merge"
                                #if n5==-1:
                                    #n5=n
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,0,255), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 3 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                                cmerge1.append(cp1)
                                cmerge2.append(cp2)
                                mergecount=mergecount+1
                                c=1
                                if check==0: ### update and swap the centers only the first time
                                    swap=1
                                    cpnew11.append(cp11[count-1])
                                    cpnew22.append(cp22[count-1])
                                    cp11.append(cpnew11[newcount-1])
                                    cp22.append(cpnew22[newcount-1])
                                    count=count+1
                                    newcount=newcount+1
                                    #connum=3
                    else:  ### there are many conotours
                       # print "Check"
                        two=1
                        notwo=1
                        track_start=1
                        enterloop=1
                        bound_rect = cv.BoundingRect(list(contour)) # calculates the up-right bounding rect for the given point set.
                        pt1 = (bound_rect[0], bound_rect[1])
                        pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                        points.append(pt1)
                        points.append(pt2)
                        cp1 = bound_rect[0] + (bound_rect[2]/2)
                        cp2 = bound_rect[1] + (bound_rect[3]/2)
                        if swap==1: ### for the next two contours after swap, keeping track of the centers
                           # print "check"
                            kk=kk+1
                            ck=cp1-cp11[count-1]
                            cj=cp1-cpnew11[newcount-1]
                            if ck<0:
                                ck=0-ck
                            if cj<0:
                                cj=0-cj
                            if ck-cj>0:
                                #if swap==1:
                                 #   n6=n5+1
                                #n6=n
                                cpnew11.append(cp1)
                                cpnew22.append(cp2)
                                newcount=newcount+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                                b=1
                            else:
                                
                                cp11.append(cp1)
                                cp22.append(cp2)
                                count=count+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                                a=1
                            if kk==2:
                                swap=0
                                
                        
                        elif count>0 and newcount>0: ### after tracking for sometime when both red and green are present
                           # print "check"
                            ck=cp1-cp11[count-1]
                            cj=cp1-cpnew11[newcount-1]
                            if ck<0:
                                ck=0-ck
                            if cj<0:
                                cj=0-cj
                            if ck-cj>0:
                                #if n8==-1:
                                 #   n8=n
                                cpnew11.append(cp1)
                                cpnew22.append(cp2)
                                newcount=newcount+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                                #print"Green", cp1,cpnew11[newcount-2]
                                b=1
                            else:

                                cp11.append(cp1)
                                cp22.append(cp2)
                                count=count+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                    #            print "Red",cp1,cp11[count-1]
                                a=1
                        elif count>0 :  ###for the first few frames when green has not yet appeared.
                            #print "check"
                            if(cp1-cp11[count-1] <10 and cp1-cp11[count-1]>-15):
                                
                                cp11.append(cp1)
                                cp22.append(cp2)
                                count=count+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                                a=1
                        
                            else:
                                
                                b=1
                                cpnew11.append(cp1)
                                cpnew22.append(cp2)
                                newcount=newcount+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                        else:   ### for the first few frames when red has not yet appeared
                                #print "check"
                               
                                cp11.append(cp1)
                                cp22.append(cp2)
                                a=1
                                count=count+1
                                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                        
                        
                            
                            
                        
                contour = contour.h_next() #to iterate through all countours.
            if track_start==1:
               #print a,b,c
               if a==1 or c==1:
                    if a==1:
                       cpk=cp11[count-1]
                       cpj=cp22[count-1]
                       a=0
                    if c==1:
                        c=0
                        cpk=cmerge1[mergecount-1]
                        cpj=cmerge2[mergecount-1]
                    #print "check"      
                    kalman.state_pre[0,0]  = cpk
                    kalman.state_pre[1,0]  = cpj
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
                    kalman_measurement[0, 0] = cpk
                    kalman_measurement[1, 0] = cpj
                    kalmancount=kalmancount+1
                
               if b==1:
                    b=0
                    #print "2"
                    kalman1.state_pre[0,0]  = cpnew11[newcount-1]
                    kalman1.state_pre[1,0]  = cpnew22[newcount-1]
                    kalman1.state_pre[2,0]  = 0
                    kalman1.state_pre[3,0]  = 0
            #print kalman.state_pre[0,0]

            # set kalman transition matrix
                    kalman1.transition_matrix[0,0] = 1
                    kalman1.transition_matrix[0,1] = 0
                    kalman1.transition_matrix[0,2] = 0
                    kalman1.transition_matrix[0,3] = 0
                    kalman1.transition_matrix[1,0] = 0
                    kalman1.transition_matrix[1,1] = 1
                    kalman1.transition_matrix[1,2] = 0
                    kalman1.transition_matrix[1,3] = 0
                    kalman1.transition_matrix[2,0] = 0
                    kalman1.transition_matrix[2,1] = 0
                    kalman1.transition_matrix[2,2] = 0
                    kalman1.transition_matrix[2,3] = 1
                    kalman1.transition_matrix[3,0] = 0
                    kalman1.transition_matrix[3,1] = 0
                    kalman1.transition_matrix[3,2] = 0
                    kalman1.transition_matrix[3,3] = 1
 
                # set Kalman Filter
                    cv.SetIdentity(kalman1.measurement_matrix, cv.RealScalar(1))
                    cv.SetIdentity(kalman1.process_noise_cov, cv.RealScalar(1e-5))
                    cv.SetIdentity(kalman1.measurement_noise_cov, cv.RealScalar(1e-1))
                    cv.SetIdentity(kalman1.error_cov_post, cv.RealScalar(1))
 
                #Prediction
                    kalman_prediction = cv.KalmanPredict(kalman1)
                    predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
                    predict_pt2.append(predict_pt)
                    #print "Prediction",predict_pt
                #Correction
                    kalman_estimated = cv.KalmanCorrect(kalman1, kalman_measurement1)
                    state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])

                #measurement
                    kalman_measurement1[0, 0] = cpnew11[newcount-1]
                    kalman_measurement1[1, 0] = cpnew22[newcount-1]
                    kalmancount1=kalmancount1+1
                    
            i=0
            while(i<kalmancount):
                pp1=predict_pt1[i]
                cv.Circle(color_image, (pp1[0]+10,pp1[1]+10), 1, cv.CV_RGB(0, 0, 0), 8)
                i=i+1
            i=0
            while(i<kalmancount1):
                pp2=predict_pt2[i]
                cv.Circle(color_image, (pp2[0]+10,pp2[1]+10), 1, cv.CV_RGB(255, 255, 255), 8)
                i=i+1
            i=0
            while(i<count):
                   cv.Circle(color_image, (cp11[i], cp22[i]), 1, cv.CV_RGB(255, 0, 0), 5)
                   i=i+1
            i=0
            while(i<newcount):
                cv.Circle(color_image, (cpnew11[i], cpnew22[i]), 1, cv.CV_RGB(0, 255, 0), 5)
                i=i+1
            i=0
            while(i<mergecount):
                cv.Circle(color_image, (cmerge1[i], cmerge2[i]), 1, cv.CV_RGB(0, 0, 255), 5)
                i=i+1
                
                
            #if(n==5):
             #   print cp11[2],cp22[2]
                    #cv.Circle(color_image, center_point1[i], 1, cv.CV_RGB(255, 0, 0), 3)
                
            cv.ShowImage("Target", color_image)
            cv.WriteFrame(writer,color_image)
            if enterloop==1:
                enterloop=0
            # Listen for ESC key
            c = cv.WaitKey(int(fps))  # to have a wait time between the displayed frames. Should be there after every showimage call
            if c == 27:  #  in case escape was pressed to close window. Hexcoded as 27, I think.
                break

if __name__=="__main__":
    t = Target()
    t.run()
