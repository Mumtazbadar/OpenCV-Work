# Loading Pre-Trained Models

Make sure to click the button below before you get started to source the correct environment.

In this exercise, you'll work to download and load a few of the pre-trained models available in the OpenVINO toolkit.

# First, you can navigate to the Pre-Trained Models list in a separate window or tab, as well as the page that gives all of the model names here.

Your task here is to download the below three pre-trained models using the Model Downloader tool, as detailed on the same page as the different model names. Note that you do not need to download all of the available pre-trained models - doing so would cause your workspace to crash, as the workspace will limit you to 3 GB of downloaded models.

# Task 1 - Find the Right Models

Using the Pre-Trained Model list, determine which models could accomplish the following tasks (there may be some room here in determining which model to download):

# Human Pose Estimation
# Text Detection
# Determining Car Type & Color
 
# Task 2 - Download the Models

Once you have determined which model best relates to the above tasks, use the Model Downloader tool to download them into the workspace for the following precision levels:

Human Pose Estimation: All precision levels
Text Detection: FP16 only
Determining Car Type & Color: INT8 only

# Note: When downloading the models in the workspace, add the -o argument (along with any other necessary arguments) with /home/workspace as the output directory. The default download directory will not allow the files to be written there within the workspace, as it is a read-only directory.

# Task 3 - Verify the Downloads
You can verify the download of these models by navigating to: /home/workspace/intel (if you followed the above note), and checking whether a directory was created for each of the three models, with included subdirectories for each precision, with respective .bin and .xml for each model.

Hint: Use the -h command with the Model Downloader tool if you need to check out the possible arguments to include when downloading specific models and precisions.

Preprocessing Inputs
Make sure to click the button below before you get started to source the correct environment.

Now that we have a few pre-trained models downloaded, it's time to preprocess the inputs to match what each of the models expects as their input. We'll use the same models as before as a basis for determining the preprocessing necessary for each input file.

As a reminder, our three models are:

Human Pose Estimation: human-pose-estimation-0001
Text Detection: text-detection-0004
Determining Car Type & Color: vehicle-attributes-recognition-barrier-0039
Note: For ease of use, these models have been added into the /home/workspace/models directory. For example, if you need to use the Text Detection model, you could find it at:

/home/workspace/models/text_detection_0004.xml
Each link above contains the documentation for the related model. In our case, we want to focus on the Inputs section of the page, wherein important information regarding the input shape, order of the shape (such as color channel first or last), and the order of the color channels, is included.

Your task is to fill out the code in three functions within preprocess_inputs.py, one for each of the three models. We have also included a potential sample image for each of the three models, that will be used with test.py to check whether the input for each model has been adjusted as expected for proper model input.

Note that each image is currently loaded as BGR with H, W, C order in the test.py file, so any necessary preprocessing to change that should occur in your three work files. Note that BGR order is used, as the OpenCV function we use to read images loads as BGR, and not RGB.

When finished, you should be able to run the test.py file and pass all three tests.

# Deploy Your First Edge App
Make sure to click the button below before you get started to source the correct environment.

So far, you've downloaded some pre-trained models, handled their inputs, and learned how to handle outputs. In this exercise, you'll implement the handling of the outputs of our three models from before, and get to see inference actually performed by adding these models to some example edge applications.

There's a lot of code still involved behind the scenes here. With the Pre-Trained Models available with the OpenVINO toolkit, you don't need to worry about the Model Optimizer, but there is still work done to load the model into the Inference Engine. We won't learn about this code until later, so in this case, you'll just need to call your functions to handle the input and output of the model within the app.

If you do want a sneak preview of some of the code that interfaces with the Inference Engine, you can check it out in inference.py. You'll work out of the handle_models.py file, as well as adding functions calls within the edge app in app.py.

# TODOs
In handle_models.py, you will need to implement handle_pose, handle_text, and handle_car.

In app.py, first, you'll need to use the input shape of the network to call the preprocessing function. Then, you need to call handle_output with the appropriate model argument in order to get the right handling function. With that function, you can then feed the output of the inference request in in order to extract the output.

Note that there is some additional post-processing done for you in create_output_image within app.py to help display the output back onto the input image.

# Testing the apps

To test your implementations, you can use app.py to run each edge application, with the following arguments:

-t: The model type, which should be one of "POSE", "TEXT", or "CAR_META"
-m: The location of the model .xml file
-i: The location of the input image used for testing
-c: A CPU extension file, if applicable. See below for what this is for the workspace. The results of your output will be saved down for viewing in the outputs directory.
As an example, here is an example of running the app with related arguments:

python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
Model Documentation
Once again, here are the links to the models, so you can use the Output section to help you get started (there are additional comments in the code to assist):

Human Pose Estimation: human-pose-estimation-0001
Text Detection: text-detection-0004
Determining Car Type & Color: vehicle-attributes-recognition-barrier-0039
Convert a TensorFlow Model
Make sure to click the button below before you get started to source the correct environment.

In this exercise, you'll convert a TensorFlow Model from the Object Detection Model Zoo into an Intermediate Representation using the Model Optimizer.

As noted in the related documentation, there is a difference in method when using a frozen graph vs. an unfrozen graph. Since freezing a graph is a TensorFlow-based function and not one specific to OpenVINO itself, in this exercise, you will only need to work with a frozen graph. However, I encourage you to try to freeze and load an unfrozen model on your own as well.

For this exercise, first download the SSD MobileNet V2 COCO model from here. Use the tar -xvf command with the downloaded file to unpack it.

From there, find the Convert a TensorFlow* Model header in the documentation, and feed in the downloaded SSD MobileNet V2 COCO model's .pb file.

If the conversion is successful, the terminal should let you know that it generated an IR model. The locations of the .xml and .bin files, as well as execution time of the Model Optimizer, will also be output.

# Note: Converting the TF model will take a little over one minute in the workspace.

Hints & Troubleshooting
Make sure to pay attention to the note in this section regarding the --reverse_input_channels argument. If you are unsure about this argument, you can read more here.

There is additional documentation specific to converting models from TensorFlow's Object Detection Zoo here. You will likely need both the --tensorflow_use_custom_operations_config and --tensorflow_object_detection_api_pipeline_config arguments fed with their related files.

# Convert a Caffe Model
Make sure to click the button below before you get started to source the correct environment.

In this exercise, you'll convert a Caffe Model into an Intermediate Representation using the Model Optimizer. You can find the related documentation here.

For this exercise, first download the SqueezeNet V1.1 model by cloning this repository.

Follow the documentation above and feed in the Caffe model to the Model Optimizer.

If the conversion is successful, the terminal should let you know that it generated an IR model. The locations of the .xml and .bin files, as well as execution time of the Model Optimizer, will also be output.

# Hints & Troubleshooting
You will need to specify --input_proto if the .prototxt file is not named the same as the model.

There is an important note in the documentation after the section Supported Topologies regarding Caffe models trained on ImageNet. If you notice poor performance in inference, you may need to specify mean and scale values in your arguments.

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
Convert an ONNX Model
Make sure to click the button below before you get started to source the correct environment.

#  Exercise Instructions
In this exercise, you'll convert an ONNX Model into an Intermediate Representation using the Model Optimizer. You can find the related documentation here.

For this exercise, first download the bvlc_alexnet model from here. Use the tar -xvf command with the downloaded file to unpack it.

Follow the documentation above and feed in the ONNX model to the Model Optimizer.

If the conversion is successful, the terminal should let you know that it generated an IR model. The locations of the .xml and .bin files, as well as execution time of the Model Optimizer, will also be output.

# PyTorch models
Note that we will only cover converting directly from an ONNX model here. If you are interested in converting a PyTorch model using ONNX for use with OpenVINO, check out this link for the steps to do so. From there, you can follow the steps in the rest of this exercise once you have an ONNX model.

# Custom Layers
Make sure to click the button below before you get started to source the correct environment.

This exercise is adapted from this repository.

Note that the classroom workspace is running OpenVINO 2019.r3, while this exercise was originally created for 2019.r2. This exercise will work appropriately in the workspace, but there may be some other differences you need to account for if you use a custom layer yourself.

The below steps will walk you through the full walkthrough of creating a custom layer; as such, there is not a related solution video. Note that custom layers is an advanced topic, and one that is not expected to be used often (if at all) in most use cases of the OpenVINO toolkit. This exercise is meant to introduce you to the concept, but you won't need to use it again in the rest of this course.

Example Custom Layer: The Hyperbolic Cosine (cosh) Function
We will follow the steps involved for implementing a custom layer using the simple hyperbolic cosine (cosh) function. The cosh function is mathematically calculated as:

cosh(x) = (e^x + e^-x) / 2
As a function that calculates a value for the given value x, the cosh function is very simple when compared to most custom layers. Though the cosh function may not represent a "real" custom layer, it serves the purpose of this tutorial as an example for working through the steps for implementing a custom layer.

Move to the next page to continue.

# Build the Model
First, export the below paths to shorten some of what you need to enter later:

export CLWS=/home/workspace/cl_tutorial
export CLT=$CLWS/OpenVINO-Custom-Layers
Then run the following to create the TensorFlow model including the cosh layer.

mkdir $CLWS/tf_model
python $CLT/create_tf_model/build_cosh_model.py $CLWS/tf_model
You should receive a message similar to:

Model saved in path: /tf_model/model.ckpt
Creating the cosh Custom Layer
Generate the Extension Template Files Using the Model Extension Generator
We will use the Model Extension Generator tool to automatically create templates for all the extensions needed by the Model Optimizer to convert and the Inference Engine to execute the custom layer. The extension template files will be partially replaced by Python and C++ code to implement the functionality of cosh as needed by the different tools. To create the four extensions for the cosh custom layer, we run the Model Extension Generator with the following options:

--mo-tf-ext = Generate a template for a Model Optimizer TensorFlow extractor
--mo-op = Generate a template for a Model Optimizer custom layer operation
--ie-cpu-ext = Generate a template for an Inference Engine CPU extension
--ie-gpu-ext = Generate a template for an Inference Engine GPU extension
--output_dir = set the output directory. Here we are using $CLWS/cl_cosh as the target directory to store the output from the Model Extension Generator.
To create the four extension templates for the cosh custom layer, given we are in the $CLWS directory, we run the command:

mkdir cl_cosh
python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=$CLWS/cl_cosh
The Model Extension Generator will start in interactive mode and prompt us with questions about the custom layer to be generated. Use the text between the []'s to answer each of the Model Extension Generator questions as follows:

Enter layer name: 
[cosh]

Do you want to automatically parse all parameters from the model file? (y/n)
...
[n]

Enter all parameters in the following format:
...
Enter 'q' when finished:
[q]

Do you want to change any answer (y/n) ? Default 'no'
[n]

Do you want to use the layer name as the operation name? (y/n)
[y]

Does your operation change shape? (y/n)  
[n]

Do you want to change any answer (y/n) ? Default 'no'
[n]
When complete, the output text will appear similar to:

Stub file for TensorFlow Model Optimizer extractor is in /home/<user>/cl_tutorial/cl_cosh/user_mo_extensions/front/tf folder
Stub file for the Model Optimizer operation is in /home/<user>/cl_tutorial/cl_cosh/user_mo_extensions/ops folder
Stub files for the Inference Engine CPU extension are in /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/cpu folder
Stub files for the Inference Engine GPU extension are in /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/gpu folder
Template files (containing source code stubs) that may need to be edited have just been created in the following locations:

TensorFlow Model Optimizer extractor extension:
$CLWS/cl_cosh/user_mo_extensions/front/tf/
cosh_ext.py
Model Optimizer operation extension:
$CLWS/cl_cosh/user_mo_extensions/ops
cosh.py
Inference Engine CPU extension:
$CLWS/cl_cosh/user_ie_extensions/cpu
ext_cosh.cpp
CMakeLists.txt
Inference Engine GPU extension:
$CLWS/cl_cosh/user_ie_extensions/gpu
cosh_kernel.cl
cosh_kernel.xml
Instructions on editing the template files are provided in later parts of this tutorial.
For reference, or to copy to make the changes quicker, pre-edited template files are provided by the tutorial in the $CLT directory.

he next page to continue.

Using Model Optimizer to Generate IR Files Containing the Custom Layer
We will now use the generated extractor and operation extensions with the Model Optimizer to generate the model IR files needed by the Inference Engine. The steps covered are:

Edit the extractor extension template file (already done - we will review it here)
Edit the operation extension template file (already done - we will review it here)
Generate the Model IR Files
Edit the Extractor Extension Template File
For the cosh custom layer, the generated extractor extension does not need to be modified because the layer parameters are used without modification. Below is a walkthrough of the Python code for the extractor extension that appears in the file $CLWS/cl_cosh/user_mo_extensions/front/tf/cosh_ext.py.

Using the text editor, open the extractor extension source file $CLWS/cl_cosh/user_mo_extensions/front/tf/cosh_ext.py.
The class is defined with the unique name coshFrontExtractor that inherits from the base extractor FrontExtractorOp class. The class variable op is set to the name of the layer operation and enabled is set to tell the Model Optimizer to use (True) or exclude (False) the layer during processing.

class coshFrontExtractor(FrontExtractorOp):
     op = 'cosh' 
     enabled = True
The extract function is overridden to allow modifications while extracting parameters from layers within the input model.

@staticmethod
 def extract(node):
The layer parameters are extracted from the input model and stored in param. This is where the layer parameters in param may be retrieved and used as needed. For the cosh custom layer, the op attribute is simply set to the name of the operation extension used.

proto_layer = node.pb
 param = proto_layer.attr
 # extracting parameters from TensorFlow layer and prepare them for IR
 attrs = {
     'op': __class__.op
 }
The attributes for the specific node are updated. This is where we can modify or create attributes in attrs before updating node with the results and the enabled class variable is returned.

# update the attributes of the node
 Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)

 return __class__.enabled
Edit the Operation Extension Template File
For the cosh custom layer, the generated operation extension does not need to be modified because the shape (i.e., dimensions) of the layer output is the same as the input shape.
Below is a walkthrough of the Python code for the operation extension that appears in the file $CLWS/cl_cosh/user_mo_extensions/ops/cosh.py.

Using the text editor, open the operation extension source file $CLWS/cl_cosh/user_mo_extensions/ops/cosh.py
The class is defined with the unique name coshOp that inherits from the base operation Op class. The class variable op is set to 'cosh', the name of the layer operation.

class coshOp(Op):
 op = 'cosh'
The coshOp class initializer __init__ function will be called for each layer created. The initializer must initialize the super class Op by passing the graph and attrs arguments along with a dictionary of the mandatory properties for the cosh operation layer that define the type (type), operation (op), and inference function (infer). This is where any other initialization needed by the coshOP operation can be specified.

def __init__(self, graph, attrs):
     mandatory_props = dict(
         type=__class__.op,
         op=__class__.op,
         infer=coshOp.infer            
     )
 super().__init__(graph, mandatory_props, attrs)
The infer function is defined to provide the Model Optimizer information on a layer, specifically returning the shape of the layer output for each node. Here, the layer output shape is the same as the input and the value of the helper function copy_shape_infer(node) is returned.

@staticmethod
 def infer(node: Node):
     # ==========================================================
     # You should add your shape calculation implementation here
     # If a layer input shape is different to the output one
     # it means that it changes shape and you need to implement
     # it on your own. Otherwise, use copy_shape_infer(node).
     # ==========================================================
     return copy_shape_infer(node)
Generate the Model IR Files
With the extensions now complete, we use the Model Optimizer to convert and optimize the example TensorFlow model into IR files that will run inference using the Inference Engine.
To create the IR files, we run the Model Optimizer for TensorFlow mo_tf.py with the following options:

--input_meta_graph model.ckpt.meta

Specifies the model input file.
--batch 1

Explicitly sets the batch size to 1 because the example model has an input dimension of "-1".
TensorFlow allows "-1" as a variable indicating "to be filled in later", however the Model Optimizer requires explicit information for the optimization process.
--output "ModCosh/Activation_8/softmax_output"

The full name of the final output layer of the model.
--extensions $CLWS/cl_cosh/user_mo_extensions

Location of the extractor and operation extensions for the custom layer to be used by the Model Optimizer during model extraction and optimization.
--output_dir $CLWS/cl_ext_cosh

Location to write the output IR files.
To create the model IR files that will include the cosh custom layer, we run the commands:

cd $CLWS/tf_model
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions $CLWS/cl_cosh/user_mo_extensions --output_dir $CLWS/cl_ext_cosh
The output will appear similar to:

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/<user>/cl_tutorial/cl_ext_cosh/model.ckpt.xml
[ SUCCESS ] BIN file: /home/<user>/cl_tutorial/cl_ext_cosh/model.ckpt.bin
[ SUCCESS ] Total execution time: x.xx seconds.
Move to the next page to continue.

Inference Engine Custom Layer Implementation for the Intel® CPU
We will now use the generated CPU extension with the Inference Engine to execute the custom layer on the CPU. The steps are:

Edit the CPU extension template files.
Compile the CPU extension library.
Execute the Model with the custom layer.
You will need to make the changes in this section to the related files.

Note that the classroom workspace only has an Intel CPU available, so we will not perform the necessary steps for GPU usage with the Inference Engine.

Edit the CPU Extension Template Files
The generated CPU extension includes the template file ext_cosh.cpp that must be edited to fill-in the functionality of the cosh custom layer for execution by the Inference Engine.
We also need to edit the CMakeLists.txt file to add any header file or library dependencies required to compile the CPU extension. In the next sections, we will walk through and edit these files.

Edit ext_cosh.cpp
We will now edit the ext_cosh.cpp by walking through the code and making the necessary changes for the cosh custom layer along the way.

Using the text editor, open the CPU extension source file $CLWS/cl_cosh/user_ie_extensions/cpu/ext_cosh.cpp.

To implement the cosh function to efficiently execute in parallel, the code will use the parallel processing supported by the Inference Engine through the use of the Intel® Threading Building Blocks library. To use the library, at the top we must include the header ie_parallel.hpp file by adding the #include line as shown below.

Before:

#include "ext_base.hpp"
 #include <cmath>
After:

#include "ext_base.hpp"
 #include "ie_parallel.hpp"
 #include <cmath>
The class coshImp implements the cosh custom layer and inherits from the extension layer base class ExtLayerBase.

class coshImpl: public ExtLayerBase {
     public:
The coshImpl constructor is passed the layer object that it is associated with to provide access to any layer parameters that may be needed when implementing the specific instance of the custom layer.

explicit coshImpl(const CNNLayer* layer) {
   try {
     ...
The coshImpl constructor configures the input and output data layout for the custom layer by calling addConfig(). In the template file, the line is commented-out and we will replace it to indicate that layer uses DataConfigurator(ConfLayout::PLN) (plain or linear) data for both input and output.

Before:

...
 // addConfig({DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
After:

addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
The construct is now complete, catching and reporting certain exceptions that may have been thrown before exiting.

} catch (InferenceEngine::details::InferenceEngineException &ex) {
     errorMsg = ex.what();
   }
 }
The execute method is overridden to implement the functionality of the cosh custom layer. The inputs and outputs are the data buffers passed as Blob objects. The template file will simply return NOT_IMPLEMENTED by default. To calculate the cosh custom layer, we will replace the execute method with the code needed to calculate the cosh function in parallel using the parallel_for3d function.

Before:

StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
     ResponseDesc *resp) noexcept override {
     // Add here implementation for layer inference
     // Examples of implementations you can find in Inference Engine tool samples/extensions folder
     return NOT_IMPLEMENTED;
After:

StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
     ResponseDesc *resp) noexcept override {
     // Add implementation for layer inference here
     // Examples of implementations are in OpenVINO samples/extensions folder

     // Get pointers to source and destination buffers
     float* src_data = inputs[0]->buffer();
     float* dst_data = outputs[0]->buffer();

     // Get the dimensions from the input (output dimensions are the same)
     SizeVector dims = inputs[0]->getTensorDesc().getDims();

     // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
     int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
     int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
     int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
     int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

     // Perform (in parallel) the hyperbolic cosine given by: 
     //    cosh(x) = (e^x + e^-x)/2
     parallel_for3d(N, C, H, [&](int b, int c, int h) {
     // Fill output_sequences with -1
     for (size_t ii = 0; ii < b*c; ii++) {
       dst_data[ii] = (exp(src_data[ii]) + exp(-src_data[ii]))/2;
     }
   });
 return OK;
 }
Edit CMakeLists.txt
Because the implementation of the cosh custom layer makes use of the parallel processing supported by the Inference Engine, we need to add the Intel® Threading Building Blocks dependency to CMakeLists.txt before compiling. We will add paths to the header and library files and add the Intel® Threading Building Blocks library to the list of link libraries. We will also rename the .so.

Using the text editor, open the CPU extension CMake file $CLWS/cl_cosh/user_ie_extensions/cpu/CMakeLists.txt.
At the top, rename the TARGET_NAME so that the compiled library is named libcosh_cpu_extension.so:

Before:

set(TARGET_NAME "user_cpu_extension")
After:

set(TARGET_NAME "cosh_cpu_extension")
We modify the include_directories to add the header include path for the Intel® Threading Building Blocks library located in /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include:

Before:

include_directories (PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/common
 ${InferenceEngine_INCLUDE_DIRS}
 )
After:

include_directories (PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/common
 ${InferenceEngine_INCLUDE_DIRS}
 "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include"
 )
We add the link_directories with the path to the Intel® Threading Building Blocks library binaries at /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:

Before:

...
 #enable_omp()
After:

...
 link_directories(
 "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib"
 )
 #enable_omp()
Finally, we add the Intel® Threading Building Blocks library tbb to the list of link libraries in target_link_libraries:

Before:

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
After:

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib} tbb)
Compile the Extension Library
To run the custom layer on the CPU during inference, the edited extension C++ source code must be compiled to create a .so shared library used by the Inference Engine. In the following steps, we will now compile the extension C++ library.

First, we run the following commands to use CMake to setup for compiling:

cd $CLWS/cl_cosh/user_ie_extensions/cpu
 mkdir -p build
 cd build
 cmake ..
The output will appear similar to:

 -- Generating done
 -- Build files have been written to: /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/cpu/build
The CPU extension library is now ready to be compiled. Compile the library using the command:

make -j $(nproc)
The output will appear similar to:

 [100%] Linking CXX shared library libcosh_cpu_extension.so
 [100%] Built target cosh_cpu_extension
Move to the next page to continue.

Execute the Model with the Custom Layer
Using a C++ Sample
To start on a C++ sample, we first need to build the C++ samples for use with the Inference Engine:

cd /opt/intel/openvino/deployment_tools/inference_engine/samples/
./build_samples.sh
This will take a few minutes to compile all of the samples.

Next, we will try running the C++ sample without including the cosh extension library to see the error describing the unsupported cosh operation using the command:

~/inference_engine_samples_build/intel64/Release/classification_sample_async -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU
The error output will be similar to:

[ ERROR ] Unsupported primitive of type: cosh name: ModCosh/cosh/Cosh
We will now run the command again, this time with the cosh extension library specified using the -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so option in the command:

~/inference_engine_samples_build/intel64/Release/classification_sample_async -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so
The output will appear similar to:

Image /home/<user>/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

classid probability
------- -----------
0       0.9308984  
1       0.0691015

total inference time: xx.xxxxxxx
Average running time of one iteration: xx.xxxxxxx ms

Throughput: xx.xxxxxxx FPS

[ INFO ] Execution successful
Using a Python Sample
First, we will try running the Python sample without including the cosh extension library to see the error describing the unsupported cosh operation using the command:

python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU
The error output will be similar to:

[ INFO ] Loading network files:
/home/<user>/cl_tutorial/tf_model/model.ckpt.xml
/home/<user>/cl_tutorial/tf_model/model.ckpt.bin
[ ERROR ] Following layers are not supported by the plugin for specified device CPU:
ModCosh/cosh/Cosh, ModCosh/cosh_1/Cosh, ModCosh/cosh_2/Cosh
[ ERROR ] Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument
We will now run the command again, this time with the cosh extension library specified using the -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so option in the command:

python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so -d CPU
The output will appear similar to:

Image /home/<user>/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

classid probability
------- -----------
0      0.9308984
1      0.0691015
Congratulations! You have now implemented a custom layer with the Intel® Distribution of OpenVINO™ Toolkit.

Feed an IR to the Inference Engine
Make sure to click the button below before you get started to source the correct environment.

Earlier in the course, you were focused on working with the Intermediate Representation (IR) models themselves, while mostly glossing over the use of the actual Inference Engine with the model.

Here, you'll import the Python wrapper for the Inference Engine (IE), and practice using different IRs with it. You will first add each IR as an IENetwork, and check whether the layers of that network are supported by the classroom CPU.

Since the classroom workspace is using an Intel CPU, you will also need to add a CPU extension to the IECore.

Once you have verified all layers are supported (when the CPU extension is added), you will load the given model into the Inference Engine.

Note that the .xml file of the IR should be given as an argument when running the script.

To test your implementation, you should be able to successfully load each of the three IR model files we have been working with throughout the course so far, which you can find in the /home/workspace/models directory.

Inference Requests
Make sure to click the button below before you get started to source the correct environment.

In the previous exercise, you loaded Intermediate Representations (IRs) into the Inference Engine. Now that we've covered some of the topics around requests, including the difference between synchronous and asynchronous requests, you'll add additional code to make inference requests to the Inference Engine.

Given an ExecutableNetwork that is the IR loaded into the Inference Engine, your task is to:

Perform a synchronous request
Start an asynchronous request given an input image frame
Wait for the asynchronous request to complete
Note that we'll cover handling the results of the request shortly, so you don't need to worry about that just yet. This will get you practice with both types of requests with the Inference Engine.

You will perform the above tasks within inference.py. This will take three arguments, one for the model, one for the test image, and the last for what type of inference request should be made.

You can use test.py afterward to verify your code successfully makes inference requests.

Integrate the Inference Engine in An Edge App
Make sure to click the button below before you get started to source the correct environment.

You've come a long way from the first lesson where most of the code for working with the OpenVINO toolkit was happening in the background. You worked with pre-trained models, moved up to converting any trained model to an Intermediate Representation with the Model Optimizer, and even got the model loaded into the Inference Engine and began making inference requests.

In this final exercise of this lesson, you'll close off the OpenVINO workflow by extracting the results of the inference request, and then integrating the Inference Engine into an existing application. You'll still be given some of the overall application infrastructure, as more that of will come in the next lesson, but all of that is outside of OpenVINO itself.

You will also add code allowing you to try out various confidence thresholds with the model, as well as changing the visual look of the output, like bounding box colors.

Now, it's up to you which exact model you want to use here, although you are able to just re-use the model you converted with TensorFlow before for an easy bounding box dectector.

Note that this application will run with a video instead of just images like we've done before.

So, your tasks are to:

Convert a bounding box model to an IR with the Model Optimizer.
Pre-process the model as necessary.
Use an async request to perform inference on each video frame.
Extract the results from the inference request.
Add code to make the requests and feed back the results within the application.
Perform any necessary post-processing steps to get the bounding boxes.
Add a command line argument to allow for different confidence thresholds for the model.
Add a command line argument to allow for different bounding box colors for the output.
Correctly utilize the command line arguments in #3 and #4 within the application.
When you are done, feed your model to app.py, and it will generate out.mp4, which you can download and view. Note that this app will take a little bit longer to run. Also, if you need to re-run inference, delete the out.mp4 file first.

You only need to feed the model with -m before adding the customization; you should set defaults for any additional arguments you add for the color and confidence so that the user does not always need to specify them.

python app.py -m {your-model-path.xml}
Handling Input Streams
Make sure to click the button below before you get started to source the correct environment.

It's time to really get in the think of things for running your app at the edge. Being able to appropriately handle an input stream is a big part of having a working AI or computer vision application.

In your case, you will be implementing a function that can handle camera, video or webcam data as input. While unfortunately the classroom workspace won't allow for webcam usage, you can also try that portion of your code out on your local machine if you have a webcam available.

As such, the tests here will focus on using a camera image or a video file. You will not need to perform any inference on the input frames, but you will need to do a few other image processing techniques to show you have some of the basics of OpenCV down.

Your tasks are to:

Implement a function that can handle camera image, video file or webcam inputs
Use cv2.VideoCapture() and open the capture stream
Re-size the frame to 100x100
Add Canny Edge Detection to the frame with min & max values of 100 and 200, respectively
Save down the image or video output
Close the stream and any windows at the end of the application
You won't be able to test a webcam input in the workspace unfortunately, but you can use the included video and test image to test your implementations.

Processing Model Outputs
Make sure to click the button below before you get started to source the correct environment.

Let's say you have a cat and two dogs at your house.

If both dogs are in a room together, they are best buds, and everything is going well.

If the cat and dog #1 are in a room together, they are also good friends, and everything is fine.

However, if the cat and dog #2 are in a room together, they don't get along, and you may need to either pull them apart, or at least play a pre-recorded message from your smart speaker to tell them to cut it out.

In this exercise, you'll receive a video where some combination or the cat and dogs may be in view. You also will have an IR that is able to determine which of these, if any, are on screen.

While the best model for this is likely an object detection model that can identify different breeds, I have provided you with a very basic (and overfit) model that will return three classes, one for one or less pets on screen, one for the bad combination of the cat and dog #2, and one for the fine combination of the cat and dog #1. This is within the exercise directory - model.xml.

It is up to you to add code that will print to the terminal anytime the bad combination of the cat and dog #2 are detected together. Note: It's important to consider whether you really want to output a warning every single time both pets are on-screen - is your warning helpful if it re-starts every 30th of a second, with a video at 30 fps?

Server Communications
Make sure to click the button below before you get started to source the correct environment.

In this exercise, you will practice showing off your new server communication skills for sending statistics over MQTT and images with FFMPEG.

The application itself is already built and able to perform inference, and a node server is set up for you to use. The main node server is already fully ready to receive communications from MQTT and FFMPEG. The MQTT node server is fully configured as well. Lastly, the ffserver is already configured for FFMPEG too.

The current application simply performs inference on a frame, gathers some statistics, and then continues onward to the next frame.

Tasks
Your tasks are to:

Add any code for MQTT to the project so that the node server receives the calculated stats
This includes importing the relevant Python library
Setting IP address and port
Connecting to the MQTT client
Publishing the calculated statistics to the client
Send the output frame (not the input image, but the processed output) to the ffserver
Additional Information
Note: Since you are given the MQTT Broker Server and Node Server for the UI, you need certain information to correctly configure, publish and subscribe with MQTT.

The MQTT port to use is 3001 - the classroom workspace only allows ports 3000-3009
The topics that the UI Node Server is listening to are "class" and "speedometer"
The Node Server will attempt to extract information from any JSON received from the MQTT server with the keys "class_names" and "speed"
Running the App
First, get the MQTT broker and UI installed.

cd webservice/server
npm install
When complete, cd ../ui
And again, npm install
You will need four separate terminal windows open in order to see the results. The steps below should be done in a different terminal based on number. You can open a new terminal in the workspace in the upper left (File>>New>>Terminal).

Get the MQTT broker installed and running.
cd webservice/server/node-server
node ./server.js
You should see a message that Mosca server started..
Get the UI Node Server running.
cd webservice/ui
npm run dev
After a few seconds, you should see webpack: Compiled successfully.
Start the ffserver
sudo ffserver -f ./ffmpeg/server.conf
Start the actual application.
First, you need to source the environment for OpenVINO in the new terminal:
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
To run the app, I'll give you two items to pipe in with ffmpeg here, with the rest up to you:
-video_size 1280x720
-i - http://0.0.0.0:3004/fac.ffm
Your app should begin running, and you should also see the MQTT broker server noting information getting published.

In order to view the output, click on the "Open App" button below in the workspace.
