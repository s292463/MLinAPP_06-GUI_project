�	mscz��@mscz��@!mscz��@	�ｺB�
@�ｺB�
@!�ｺB�
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLmscz��@ 8��L�?1Ps�"0@A��|y��?I�Y�h9�@YG!ɬ���?rEagerKernelExecute 0*	�Zd�n@2U
Iterator::Model::ParallelMapV21���C��?!)��f9�J@)1���C��?1)��f9�J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���ګ�?!ͫ<7��2@)H��[�J�?1��,60@:Preprocessing2F
Iterator::Model��Z��?!mz�ƚ`P@)�ڋh;��?1����?(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�L�J�?!�Xg�P� @)�L�J�?1�Xg�P� @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0עh[�?!n�0:'@)/�:�?1����
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����˵?!$yr�>A@)�W�B�?1�T [<	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����{?!�\S�b@)����{?1�\S�b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%��rޟ?!��j��6)@)���XPd?1�d�]��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�51.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�ｺB�
@ISFa��hO@Q����~�@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 8��L�? 8��L�?! 8��L�?      ��!       "	Ps�"0@Ps�"0@!Ps�"0@*      ��!       2	��|y��?��|y��?!��|y��?:	�Y�h9�@�Y�h9�@!�Y�h9�@B      ��!       J	G!ɬ���?G!ɬ���?!G!ɬ���?R      ��!       Z	G!ɬ���?G!ɬ���?!G!ɬ���?b      ��!       JGPUY�ｺB�
@b qSFa��hO@y����~�@@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��t�?!��t�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad���|�?!�9���?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�n�ܥ?!�ީ�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad=�J�ޤ?!�̑����?"1
model/Conv1D_2/conv1dConv2D��%��?!)��ŵ��?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose;�;:o��?!\O2ʨ
�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput@m+F�?!���i'�?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterq¬�
٠?!RU�B�?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�.��M�?!'�}3HL�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose7_\;�?!�`��S�?Q      Y@Y     �-@a     JU@qd}���4@y`LÆO0�?"�
both�Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�51.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�20.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 