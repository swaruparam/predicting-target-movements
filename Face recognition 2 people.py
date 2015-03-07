import cv
import cv2
import numpy as np
cascade = cv.Load('F:\OpenCV-2.1.0\data\haarcascades\haarcascade_frontalface_alt.xml')

class Target:

    def __init__(self): 
        self.capture = cv.CaptureFromFile('F:\\Project\\Video12\\straight1.mov') 
        cv.NamedWindow("Target", cv.CV_WINDOW_NORMAL)
        #cv.NamedWindow("GreyScale", cv.CV_WINDOW_NORMAL)

    def detect(self,image):
         image_faces = []
         xcord=[]
         ycord=[]
         faces = cv.HaarDetectObjects(image, cascade, cv.CreateMemStorage(0))
 
         if faces:
          for (x,y,w,h),n in faces:
           xcord.append(x)
           ycord.append(y)
           image_faces.append(image[y:(y+h), x:(x+w)])
           cv.Rectangle(image,(x,y),(x+w,y+h),(255,255,255),3)
         return image_faces,xcord,ycord

    
        
    def featureextract (self,imgg,templateg):
        #print "check"
        surfDetector = cv2.FeatureDetector_create("SURF")
        surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")

        kp = surfDetector.detect(imgg)
        kp, descritors = surfDescriptorExtractor .compute(imgg,kp)

        samples = np.array(descritors)
        responses = np.arange(len(kp),dtype = np.float32)

        knn = cv2.KNearest()
        knn.train(samples,responses)

        keys = surfDetector.detect(templateg)
        keys, desc = surfDescriptorExtractor .compute(templateg,keys)

        rowsize = len(desc) / len(keys)
        if rowsize > 1:
            hrows = np.array(desc, dtype = np.float32).reshape((-1, rowsize))
            nrows = np.array(descritors, dtype = np.float32).reshape((-1, rowsize))
        else:
            hrows = np.array(desc, dtype = np.float32)
            nrows = np.array(descritors, dtype = np.float32)
            rowsize = len(hrows[0])

        matched=0
        total=0
        for h,des in enumerate(desc):
            des = np.array(des,np.float32).reshape((1,rowsize))
            retval, results, neigh_resp, dists = knn.find_nearest(des,1)
            res,dist =  int(results[0][0]),dists[0][0]

            if dist<0.1:
                color = (0,0,255)
                matched += 1
                #total +=1
            else:  
                color = (255,0,0)
                #total+=1
            #Draw matched key points on original image
            #x,y = kp[res].pt
            #center = (int(x),int(y))
            #cv2.circle(imgg,center,2,color,-1)

            #Draw matched key points on template image
            #x,y = keys[h].pt
            #center = (int(x),int(y))
            #cv2.circle(templateg,center,2,color,-1)
        
        #perc=total-matched
        #print matched
        #cv2.namedWindow("Matched Keypoints in original",1);
        #cv2.imshow('Matched Keypoints in original',imgg)
        #cv2.namedWindow("Matched Keypoints in extracted",2);
        #cv2.imshow('Matched Keypoints in extracted',templateg)
        #print matched
        #print "check"
        #print total
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return matched

    def run(self):

        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        fourcc=cv.CV_FOURCC('X', 'V', 'I', 'D')
        fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
        writer=cv.CreateVideoWriter('output.avi',fourcc,fps,cv.GetSize(frame),1)
        
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)
        
        first = True
        second=True
        n=0
        flag1=0
        flag2=0
        el1=0
        el2=0
        track_start1=0
        track_start2=0
        count1=0
        count2=0
        ff1=ff2=0
        cp11=[]
        cp22=[]
        cp11new=[]
        cp22new=[]
        kalmancount1=0
        kalmancount2=0
        predict_pt1=[]
        predict_pt2=[]
        points=[]

        while True:
            n=n+1
            print n
            closest_to_left = cv.GetSize(frame)[0]
            closest_to_right = cv.GetSize(frame)[1]

            color_image = cv.QueryFrame(self.capture)
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)
            if first:
                    difference = cv.CloneImage(color_image) 
                    temp = cv.CloneImage(color_image)
                    cv.ConvertScale(color_image, moving_average, 1.0, 0.0) 
                    first = False 
            elif n<150:           
                    cv.RunningAvg(color_image, moving_average, 0.02, None)
            cv.ConvertScale(moving_average, temp, 1.0, 0.0)

            
            cv.AbsDiff(color_image, temp, difference) 
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY) 
            cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)
            
            
            cv.Dilate(grey_image, grey_image, None, 18)  
            cv.Erode(grey_image, grey_image, None, 10)

            cv.ShowImage("GreyScale", grey_image)
            storage = cv.CreateMemStorage(0)
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            contour1 = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            #contour2 = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

            noc=0
            q=r=-1
            q,r=divmod(n,25)

            while contour1 and n>300 and r==0:
                    area=cv.ContourArea(list(contour1))
                    el1=el2=0
                    #print area
                    if (area > 3000.0):
                        #print noc
                        noc=noc+1
                    contour1 = contour1.h_next()

            fps=cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS)
            frame_size = cv.GetSize(frame)
            width=frame_size[0]
            height=frame_size[1]
            image_faces = []
            xpos=[]
            ypos=[]
            f1=cv2.imread("F:\\Project\\Video12\\rash.jpg")
            f2=cv2.imread("F:\\Project\\Video12\\vaish.jpg")
            f3=cv2.resize(f1,(400,400))
            f4=cv2.resize(f2,(400,400))
            if noc>0 and (flag1==0 or flag2==0) and n>150:
                ##Face recognition to be performed
                #print "checkface",n
                image_faces,xpos,ypos = self.detect(color_image)
                for i, face in enumerate(image_faces):
                   cv.SaveImage("face-" + str(i) + ".jpg", face)
                   f=cv2.imread("face-"+str(i)+".jpg")
                   f5=cv2.resize(f,(400,400))
                   d1=self.featureextract(f5,f3)
                   d2=self.featureextract(f5,f4)
                   #print d1,d2
                   if d1-d2>=5.0:
                       xx1=xpos[i]
                       yy1=ypos[i]
                    #   print xx,yy
                       flag1=1
                   if d2-d1>=4.0:
                        xx2=xpos[i]
                        yy2=ypos[i]
                        flag2=1
                   #print flag1,flag2
                i=0
                k=0
                total=0
                
            while contour:
                    #print "checktrack",n
                    area=cv.ContourArea(list(contour))
                    #print area
                    if (area > 3000.0):
                        bound_rect = cv.BoundingRect(list(contour)) 
                        pt1 = (bound_rect[0], bound_rect[1])
                        pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                        points.append(pt1)
                        points.append(pt2)
                        #cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                        #font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                        #cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                        cp1 = bound_rect[0] + (bound_rect[2]/2)
                        cp2 = bound_rect[1] + (bound_rect[3]/2)
                        #if flag1==1 and flag2==1:
                         #   ff=1
                        if flag1==1:
                            if el1==0:
                                #print "check"
                                # Create Kalman Filter
                                kalman1 = cv.CreateKalman(4, 2, 0)
                                kalman_state1 = cv.CreateMat(4, 1, cv.CV_32FC1)
                                kalman_process_noise1 = cv.CreateMat(4, 1, cv.CV_32FC1)
                                kalman_measurement1 = cv.CreateMat(2, 1, cv.CV_32FC1)
                                el1=1
                            if ff1==0:
                            #print cp1,xx,bound_rect[2]/2
                                if cp1-xx1<(bound_rect[2]/2):
                                    ff1=1
                                    track_start1=1
                                    cp11.append(cp1)
                                    cp22.append(cp2)
                                    count1=count1+1
                                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                    cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                            else:
                                if cp1>cp11[count1-1]:
                                #print "check"
                                    if cp1-cp11[count1-1]<=100:
                                        track_start1=1
                                        cp11.append(cp1)
                                        cp22.append(cp2)
                                        count1=count1+1
                                        cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                        cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                                else:
                                    #print "check1"
                                    if cp11[count1-1]-cp1<=100:
                                        track_start1=1
                                        cp11.append(cp1)
                                        cp22.append(cp2)
                                        count1=count1+1
                                        cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 3)
                                        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                        cv.PutText(color_image,"contour 1 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,0,255))
                        if flag2==1:
                            if el2==0:
                                #print "check1"
                                # Create Kalman Filter
                                kalman2 = cv.CreateKalman(4, 2, 0)
                                kalman_state2 = cv.CreateMat(4, 1, cv.CV_32FC1)
                                kalman_process_noise2 = cv.CreateMat(4, 1, cv.CV_32FC1)
                                kalman_measurement2 = cv.CreateMat(2, 1, cv.CV_32FC1)
                                el2=1
                            if ff2==0:
                            #print cp1,xx,bound_rect[2]/2
                                if cp1-xx2<(bound_rect[2]/2):
                                    #print "check2"
                                    ff2=1
                                    track_start2=1
                                    cp11new.append(cp1)
                                    cp22new.append(cp2)
                                    count2=count2+1
                                    cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                    cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                            else:
                                if cp1>cp11new[count2-1]:
                                #print "check"
                                    if cp1-cp11new[count2-1]<=100:
                                        track_start2=1
                                        cp11new.append(cp1)
                                        cp22new.append(cp2)
                                        count2=count2+1
                                        cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                        cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                                else:
                                    #print "check1"
                                    if cp11new[count2-1]-cp1<=100:
                                        track_start2=1
                                        cp11new.append(cp1)
                                        cp22new.append(cp2)
                                        count2=count2+1
                                        cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(0,255,0), 3)
                                        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, 0, 3, 8)
                                        cv.PutText(color_image,"contour 2 h=""%d"%bound_rect[3],(bound_rect[0],bound_rect[1]), font,(0,255,0))
                    contour = contour.h_next()        
            if track_start1==1:
                    
                    kalman1.state_pre[0,0]  = cp11[count1-1]
                    kalman1.state_pre[1,0]  = cp22[count1-1]
                    kalman1.state_pre[2,0]  = 0
                    kalman1.state_pre[3,0]  = 0
                    #print kalman.state_pre[0,0]

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
 
                    cv.SetIdentity(kalman1.measurement_matrix, cv.RealScalar(1))
                    cv.SetIdentity(kalman1.process_noise_cov, cv.RealScalar(1e-5))
                    cv.SetIdentity(kalman1.measurement_noise_cov, cv.RealScalar(1e-1))
                    cv.SetIdentity(kalman1.error_cov_post, cv.RealScalar(1))
     
                    kalman_prediction = cv.KalmanPredict(kalman1)
                    predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
                    predict_pt1.append(predict_pt)
                    #print "Prediction",predict_pt
                
                    kalman_estimated = cv.KalmanCorrect(kalman1, kalman_measurement1)
                    state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])
    
                    kalman_measurement1[0, 0] = cp11[count1-1]
                    kalman_measurement1[1, 0] = cp22[count1-1]
                    kalmancount1=kalmancount1+1

            if track_start2==1:
                    #print cp11new[count2-1],cp22new[count2-1]
                    kalman2.state_pre[0,0]  = cp11new[count2-1]
                    kalman2.state_pre[1,0]  = cp22new[count2-1]
                    kalman2.state_pre[2,0]  = 0
                    kalman2.state_pre[3,0]  = 0
                    #print kalman.state_pre[0,0]

                    kalman2.transition_matrix[0,0] = 1
                    kalman2.transition_matrix[0,1] = 0
                    kalman2.transition_matrix[0,2] = 0
                    kalman2.transition_matrix[0,3] = 0
                    kalman2.transition_matrix[1,0] = 0
                    kalman2.transition_matrix[1,1] = 1
                    kalman2.transition_matrix[1,2] = 0
                    kalman2.transition_matrix[1,3] = 0
                    kalman2.transition_matrix[2,0] = 0
                    kalman2.transition_matrix[2,1] = 0
                    kalman2.transition_matrix[2,2] = 0
                    kalman2.transition_matrix[2,3] = 1
                    kalman2.transition_matrix[3,0] = 0
                    kalman2.transition_matrix[3,1] = 0
                    kalman2.transition_matrix[3,2] = 0
                    kalman2.transition_matrix[3,3] = 1
 
                    cv.SetIdentity(kalman2.measurement_matrix, cv.RealScalar(1))
                    cv.SetIdentity(kalman2.process_noise_cov, cv.RealScalar(1e-5))
                    cv.SetIdentity(kalman2.measurement_noise_cov, cv.RealScalar(1e-1))
                    cv.SetIdentity(kalman2.error_cov_post, cv.RealScalar(1))
     
                    kalman_prediction = cv.KalmanPredict(kalman2)
                    #print kalman_prediction[0,0]
                    predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
                    #print predict_pt
                    predict_pt2.append(predict_pt)
                    #print "Prediction",predict_pt
                
                    kalman_estimated = cv.KalmanCorrect(kalman2, kalman_measurement2)
                    state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])
    
                    kalman_measurement2[0, 0] = cp11new[count2-1]
                    kalman_measurement2[1, 0] = cp22new[count2-1]
                    kalmancount2=kalmancount2+1

            i=0
            while(i<kalmancount1):
                    #print pp1
                    pp1=predict_pt1[i]
                    cv.Circle(color_image, (pp1[0]+10,pp1[1]+10), 1, cv.CV_RGB(0, 0, 0), 8)
                    i=i+1
            i=0
            while(i<kalmancount2):
                    #print predict_pt2[i]
                    pp2=predict_pt2[i]
                    cv.Circle(color_image, (pp2[0]+10,pp2[1]+10), 1, cv.CV_RGB(255, 255, 255), 8)
                    i=i+1
            i=0
            while(i<count1):
                   #print "c",n
                   cv.Circle(color_image, (cp11[i], cp22[i]), 1, cv.CV_RGB(255, 0, 0), 5)
                   i=i+1
            i=0
            while(i<count2):
                    #print "d",n
                    cv.Circle(color_image, (cp11new[i], cp22new[i]), 1, cv.CV_RGB(0, 255, 0), 5)
                    i=i+1
            cv.ShowImage("Target", color_image)
            c = cv.WaitKey(int(fps))  # to have a wait time between the displayed frames. Should be there after every showimage call
            if c == 27:  #  in case escape was pressed to close window. Hexcoded as 27, I think.
                    break

if __name__=="__main__":
    t = Target()
    t.run()

                
                            
            
