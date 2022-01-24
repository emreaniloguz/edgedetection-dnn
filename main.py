import cv2 
import numpy as np
from imutils.video import  WebcamVideoStream


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0


    
    def getMemoryShapes(self, inputs):
        
        inputShape, targetShape = inputs[0], inputs[1]
        
        batchSize, numChannels = inputShape[0], inputShape[1]
        
        height, width = targetShape[2], targetShape[3]
        
        
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)

        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        
        self.yend = self.ystart + height
        
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]










class Processing():
    def __init__(self):

        # Load the pre-trained model
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
        
        # Specify target device
        cv2.dnn_registerLayer("Crop", CropLayer)

        # Create ORB object
        self.orb = cv2.ORB_create()

    
    def feed_dnn(self,frame):
        
        frame_input = cv2.dnn.blobFromImage(frame, scalefactor=1.0)

        # Set input to the network
        self.net.setInput(frame_input)

        # Run forward pass to get output of layer conv5-2
        self.out = self.net.forward()

        # Get scores and geometry
        self.out = self.out[0][0]

        
        self.out = cv2.resize(self.out, (frame.shape[1], frame.shape[0]))

        self.out = cv2.cvtColor(self.out, cv2.COLOR_GRAY2BGR)

        self.out = 255*self.out
        
        self.out = np.uint8(self.out)

        
        kp1, des1 = self.orb.detectAndCompute(self.out, None)

        # Draw output with keypoints
        w_kps = cv2.drawKeypoints(self.out, kp1, None, color=(0, 255, 0), flags=0)

        # Stack frames
        concatenate = np.concatenate((frame, self.out,w_kps), axis=1)

        return concatenate







class VideoInput():
    def __init__(self):
        
        self.edge_dnn = Processing()
        
        # Initialize the video stream and allow the camera sensor to warm up
        self.thread = WebcamVideoStream

        # Start the thread to read frames from the video stream
        self.thread_obj = self.thread(src=0).start()

        
    
    def scan_frame_feed(self):
        # grab the frame from the threaded video stream

        while True:
            self.frame = self.thread_obj.read()


            

            concatenated = self.edge_dnn.feed_dnn(self.frame)


            cv2.imshow("Frame", concatenated)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        
        cv2.destroyAllWindows()




if __name__ == "__main__":


    
    vs = VideoInput()
    

    vs.scan_frame_feed()