�	ŏ1w-�@ŏ1w-�@!ŏ1w-�@	hDw4VQ@hDw4VQ@!hDw4VQ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLŏ1w-�@�m��)�?1˃�9@A�h�x�J�?I��tB@Y�쟧��?rEagerKernelExecute 0*	��Q��d@2F
Iterator::Model}v�uŌ�?!�u�/�H@)��_=�[�?1'}�X7A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�1%��?!W,Ջ��9@)�j��u�?1^�{�5@:Preprocessing2U
Iterator::Model::ParallelMapV26 B\9{�?!�s��+@)6 B\9{�?1�s��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�:�zj�?!�B�)@)�:�zj�?1�B�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!��G���3@))%���?1�o@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�6�4D�?!W�#�(�I@)幾	�?1�觞�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;ŪA�{?!9�@|.@);ŪA�{?19�@|.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapͬ�����?!�wj�5@)c`�e?1�83|��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�35.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9iDw4VQ@I7�Vل
K@Q;U`P�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�m��)�?�m��)�?!�m��)�?      ��!       "	˃�9@˃�9@!˃�9@*      ��!       2	�h�x�J�?�h�x�J�?!�h�x�J�?:	��tB@��tB@!��tB@B      ��!       J	�쟧��?�쟧��?!�쟧��?R      ��!       Z	�쟧��?�쟧��?!�쟧��?b      ��!       JGPUYiDw4VQ@b q7�Vل
K@y;U`P�D@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��s���?!��s���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad+�m�§?!�
GU?�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad���]!m�?!8ɾ�X��?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradv��.Ө�?!�0_x���?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��i-��?!�Xgil�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�J���?!����/J�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose"�m"+�?!�}.G�o�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter5���*��?! ��ـ�?0"3
model/Conv1D_1/BiasAddBiasAdd��o�!�?!2�͠��?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�T�8m�?!~T;!�{�?Q      Y@Y@n]�G*@a8R4��U@q%����:@y��W��w�?"�
both�Your program is POTENTIALLY input-bound because 18.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�35.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�26.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 