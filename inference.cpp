#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <fstream>
#include <sstream>

#include <memory>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

vector<Rect_<int>> detect_faces(Net facenet, Mat frame){
  /* function  detect_faces:
    args: net, frame
    return: faces vector
  */
    float confidenceThreshold = 0.7;

    Mat inputBlob = blobFromImage(resize(frame, frame, Size(300, 300), INTER_CUBIC,
                                          1.0,(300,300),(104.0, 177.0, 123.0)));
    facenet.setInput(inputBlob);
    Mat detections = facenet.forward();

    vector<Rect_<int>> faces;
    for (int i = 0; i < detections.rows; i++){
        
        float confidence = detection.at<float>(i, 2);
        if (confidence > confidenceThreshold){

            int idx = static_cast<int>(detections.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detections.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detections.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detections.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detections.at<float>(i, 6) * frame.rows);

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            
            faces.push_back(object)
        }
    }

    return faces
}

int main(int argc, char **argv){
  
  const string keys =
        "{ h help |                                    | print this help message }"
        "{ proto  | models/deploy.prototxt.txt    | model configuration }"
        "{ model  | models/res10_300x300_ssd_iter_140000.caffemodel | model weights }";

  CommandLineParser parser(argc, argv, keys);
  if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

  // Load the Face detector weights
  string modelTxt = samples::findFile(parser.get<string>("proto"));
  string modelBin = samples::findFile(parser.get<string>("model"));
  Net facenet = dnn::readNetFromCaffe(modelTxt, modelBin);
  
  //load the torchscript model
  torch::jit::script::Module mask_model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mask_model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }

  // Create a VideoCapture object and use camera to capture the video
  VideoCapture cap(0); 

  // Check if camera opened successfully
  if(!cap.isOpened())
  {
    cout << "Error opening video stream" << endl; 
    return -1; 
  } 

  // Default resolution of the frame is obtained.The default resolution is system dependent. 
  //int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
  //int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  const int W_in = 224;
  const int H_in = 224;

  // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
  //VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(W_in,H_in)); 
  
  vector<string> labels{'No Mask','Mask'};
  vector<tuple> labelColor{CV_RGB(255, 0,0), CV_RGB(0, 255,0)};

  // Transformations part

  //

  while(1)
  { 
    Mat frame; 
    
    // Capture frame-by-frame 
    cap >> frame;
 
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    
    // Write the frame into the file 'outcpp.avi'
    //video.write(frame);

    //The Rect class holds the four borders of a rectangle: left, top, right, and bottom.
    vector<Rect_<int>> faces;
    faces = detect_faces(facenet,frame);

    for(int i = 0; i < faces.size(); i++){

      // Process face by face:
      Rect face_i = faces[i];
      Mat face = gray(face_i);
      
      Mat face_resized;
      resize(face, face_resized, Size(W_in, H_in), 1.0, 1.0, INTER_CUBIC);

      // Convert the image to a tensor.
      torch::Tensor img_tensor = torch::from_blob(face_resized.data, { 1, face_resized.rows, face_resized.cols, 3 }, torch::kByte);
      img_tensor = img_tensor.permute({ 0, 3, 1, 2 }); // convert to CxHxW
      img_tensor = img_tensor.to(torch::kFloat);

      // Now perform the prediction
      torch::Tensor outputs = mask_model.forward(img_tensor).toTensor();
      outputs = at::sigmoid(outputs);
      //convert outputs to float
      bool pred = outputs>0.5;
      
      // And finally write all we've found out to the original image!
      // First of all draw a green rectangle around the detected face:
      rectangle(frame, face_i, labelColor[pred], 1);
      
      // Create the text we will annotate the box with:
      string box_text = format("%s", labels[pred]);
      
      // Calculate the position for annotated text (make sure we don't
      // put illegal values in there):
      int pos_x = max(face_i.tl().x - 10, 0);
      int pos_y = max(face_i.tl().y - 10, 0);
      // And now put it into the image:
      putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    }
   
    // Display the resulting frame    
    imshow( "Frame", frame );
 
    // Press  ESC on keyboard to  exit
    char c = (char)waitKey(1);
    if( c == 27 ) 
      break;
  }

  // When everything done, release the video capture and write object
  cap.release();
  //video.release();

  // Closes all the windows
  destroyAllWindows();
  return 0;
}