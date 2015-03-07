import cv
import numpy
class Target:

    def __init__(self): #_init_ is the constructor, self is used like this pointer. Object to the class target. capture is a data member of the class
         #CaptureFromCAM(0) for webcam videos.
        self.capture = cv.CaptureFromFile('F:\\Project\\Video13\\traj5.mp4') 
        #cv.NamedWindow("Target", 1)
        cv.NamedWindow("Target", cv.CV_WINDOW_NORMAL)
        cv.NamedWindow("GreyScale",cv.CV_WINDOW_NORMAL)
       # cv.ResizeWindow("Target",600,600)
        
        #fps=20.0
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        print frame_size
        writer=cv.CreateVideoWriter('output.mpg',0,5,cv.GetSize(frame),1)
        #self.out=cv.CreateVideoWriter('finalvid.avi',fourcc,fps,frame_size,True)

    def run(self):
        
        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        #frame = cv.CreateMat(fr.rows/6, fr.cols/6, fr.type)
        #cv.Resize(im, frame)
        #cv.ShowImage("target",frame)
        frame_size = cv.GetSize(frame)
        h=frame_size[1]
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        writer=cv.CreateVideoWriter('output.avi',fourcc,fps,cv.GetSize(frame),1)
        ##print frame_size
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)# arguments-(width,height), bit depth - no.of bits, no. of channels per pixel
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)
        # Create Kalman Filter
        kalman = cv.CreateKalman(4, 2, 0)
        kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)
        print fps
        occlusion=1
        velocity=1
        track_start=0
        first = True
        second=True
        n=0
        cp11 = []
        cp22 = []
        center_point1 = []
        lasttenx=[]
        lastteny=[]
        predict_pt1 = []
        xmove = []
        count=0
        kalmancount=0
        enterloop=0
        f=1
        occ=0
        pltx=[]
        plty=[]
        flag=0
        n1=-1
            
        while True:
            #if n1==n:
                #print "check"
                #cv.SaveImage('22.jpg',color_image)
            n=n+1
            print n
            closest_to_left = cv.GetSize(frame)[0]
            ##print closest_to_left
            closest_to_right = cv.GetSize(frame)[1]
            ##print closest_to_right
            color_image = cv.QueryFrame(self.capture)

            # Smooth to get rid of false positives
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)# src, dst, smoothing using gaussian kernal-works with 1,3 channels. can have src,dst
                                #same. 8bit or 32 bit floating point images.

            if first:
                t=1
                #print "check"
                difference = cv.CloneImage(color_image) #fully copies the image.
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0) #sort of copying. src,dst,scale,shift-allows type conversion. value added after
                                    #scaling. Optional scaling, shifting.
                first = False # for the first frame alone.
            elif n:           #for all the other frames
                cv.RunningAvg(color_image, moving_average, 0.02, None) # second argument must be 32F or 64F.
                            # img, accumulated image-running average of prev frames, alpha - how fast previous images are forgotten, mask-optional.
            #if n>900:
             #second=False
            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1.0, 0.0) # cannot use clone because of diff in bit depths.
            if n==330:
                #print n1
                cv.SaveImage('11.jpg',temp)
            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, difference) # all three images of same type. src1-src2.

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY) # if range is going to be a problem, scale the image to the required range before using this.

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 50, 255, cv.CV_THRESH_BINARY)#src, dst, thresh, max. value, thresholdtype).
                                #Used only on greyscale images. can be used for removing noise, by changing thresholdtype.

            # Dilate and erode to get people blobs
            cv.Dilate(grey_image, grey_image, None, 18)  # src,dst- should be same type, element=Mat() can be a matrix used for dilation, No. of iterations.
        
            cv.Erode(grey_image, grey_image, None, 10) #replaces pixel with min value of all the pixels in the neighbourhood specified by the element
                                #matrix. Dilation replaces the pixel with max value.
            
            cv.ShowImage("GreyScale", grey_image)
            if n==430:
                #print "check"
                cv.SaveImage('55.jpg',grey_image)
            if n==510:
                cv.SaveImage('66.jpg',grey_image)
            if n==730:
                cv.SaveImage('77.jpg',grey_image)
            #if n==400:
             #   cv.SaveImage('88.jpg',grey_image)
            storage = cv.CreateMemStorage(0) #0 allocates default value of 64K memory bytes.
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE) # performs only on binary images.src= 8bit 1channel,
                    #countours- vectors of pts. src changed while finding countours. 3rd arg mode- gets countours and puts them in a 2 level hirarchy.
                    # 4th arg method- gets only the endpoints. if rectangular- 4 pts are extracted.

            points = []
            i=0
            k=0
            total=0
            while contour and n>150:# if contours are found.
                
                area=cv.ContourArea(list(contour))
                #bound_rect = cv.BoundingRect(list(contour))
                #h=bound_rect[3]
                #print h
                #print area
                if (area>35000.0):#for away1 6000.0 to 15000.0
                    occ=0
                    if n1==-1:
                        n1=615
                    #print n
                    #print area
                    track_start=1
                    enterloop=1
                    bound_rect = cv.BoundingRect(list(contour)) # calculates the up-right bounding rect for the given point set.
                    pt1 = (bound_rect[0], bound_rect[1])
                    pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                    points.append(pt1)
                    points.append(pt2)
                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 7)
                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                    cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                    
                    cp1 = bound_rect[0] + (bound_rect[2]/2)
                    cp2 = bound_rect[1] + (bound_rect[3]/2)
                    cp11.append(cp1)
                    cp22.append(cp2)
                    if t==1:
                       # x=cp11[count-10]
                        #y=h-cp2[count-10]
                        t=0
                    
                 
                    if count>0:
                        xdash=cp11[count]-cp11[count-1]

                        if xdash<0:
                            xdash=-cp11[count-1]+cp11[count]
                        xmove.append(xdash)
                        while(k<count):
                            #print "Stuck"
                            total=total+xmove[k]
                            k=k+1
                        if total==0:
                            velocity=6
                        else:
                            velocity=total/count
                        #print velocity
                    count=count+1
                    if count>0:
                        if cp22[count-1]-cp22[count-2]>5:
                            x=cp11[count-1]
                            y=h-cp22[count-1]
                    #kalmancount=kalmancount+1
                    #print count,kalmancount
                if count>0:
                    #lasttenx=cp11[count-3:count-1]
                    #lastteny=cp22[count-3:count-1]
                    lasttenx=cp11[count-72:count-2]
                    lastteny=cp22[count-72:count-2]
                contour = contour.h_next() #to iterate through all countours.
            if track_start==1 and enterloop==0:
                #if n1==-1:
                #n1=n
                p = numpy.poly1d(numpy.polyfit(lasttenx, lastteny, deg=2))
                #print "check"
                #print p
                if f==1:
                    kk=1
                    while kk<69:
                        #print "check",kk
                        
                        plx= lasttenx[kk]
                        ply=int(p(plx))
                        #print ply
                        pltx.append(plx)
                        plty.append(ply)
                        kk=kk+1
                    f=0
                #print len(lasttenx)
                p1=numpy.polyder(p)
                #print p,p(cp1),cp2
                cp1=cp1+3
                cp2=p(cp1)
                occ=1
                #print velocity
                #print cp1,cp2,p
                #lasttenx.append(cp1)
                #lastteny.append(cp2)
                    #print "Occlusion"
                 #   print "Occlsion",velocity
                    #if occlusion==1:
                    #print "check"
                    #x=cp11[1]
                    #y=h-cp22[1]
                    #print cp1,cp2,x,y
                    #x1=cp1
                #y1=h-cp2
                    #m=float(y1-y)/float(x1-x)
                    #occlusion=0
                    #print m
                    #if m<0:
                     #   m=0-m
                    #cp1=int(kalman_prediction[0,0])+velocity
                    #cp1=cp11[count-1]+velocity
                    #cp1=cp1+velocity
                    #cp2=h-int((kalman_prediction[0,0])*m )###Find m
                    #cp2=h-int(m*cp1) -y
                    #print cp1,cp2,
                  #  print cp1,cp2
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

                #lasttenx.append(predict_pt[0])
                #lastteny.append(predict_pt[1])
            
            while(i<kalmancount):
                #if occ==0:
                    pp1=predict_pt1[i]
                    cv.Circle(color_image, (pp1[0]-10,pp1[1]-10), 1, cv.CV_RGB(0, 255, 0), 10)
                    i=i+1
                #else:
                 #   pp1=predict_pt1[i]
                  #  cv.Circle(color_image, (pp1[0]-10,pp1[1]-10), 1, cv.CV_RGB(0, 0, 0), 10)
                   # i=i+1
                    
            i=0
            while (track_start==1 and enterloop==0 and i<kk-1) or (flag==1 and i<kk-1):
                cv.Circle(color_image, (pltx[i],plty[i]), 1, cv.CV_RGB(0,0,255),20)
                flag=1
                i=i+1
            i=0
            while(i<count):
                   cv.Circle(color_image, (cp11[i]-50, cp22[i]-50), 1, cv.CV_RGB(255, 0, 0), 10)
                   i=i+1 
            #if(n==5):
               # print cp11[2],cp22[2]
                    #cv.Circle(color_image, center_point1[i], 1, cv.CV_RGB(255, 0, 0), 3)
                
            cv.ShowImage("Target", color_image)
            if n==430:
                cv.SaveImage("11.jpg",color_image)
            if n==510:
                cv.SaveImage("22.jpg",color_image)
            if n==730:
                cv.SaveImage("33.jpg",color_image)
            #if n==400:
             #   cv.SaveImage("44.jpg",color_image)
            cv.WriteFrame(writer,color_image);
            if enterloop==1:
                enterloop=0
            # Listen for ESC key
            c = cv.WaitKey(int(fps/4))  # to have a wait time between the displayed frames. Should be there after every showimage call
            if c == 27:  #  in case escape was pressed to close window. Hexcoded as 27, I think.
                break

if __name__=="__main__":
    t = Target()
    t.run()
