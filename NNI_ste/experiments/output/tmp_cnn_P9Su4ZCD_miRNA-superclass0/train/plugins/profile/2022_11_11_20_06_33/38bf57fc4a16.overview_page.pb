�	3����7@3����7@!3����7@	�eM�i�?�eM�i�?!�eM�i�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3����7@TS�u8�.@1�$�� @A_�BF��?I�}���@Y���~�?rEagerKernelExecute 0*	�� �r4b@2U
Iterator::Model::ParallelMapV2;������?! n�{�6@);������?1 n�{�6@:Preprocessing2F
Iterator::Model�~�T�°?!#�3)zF@)5�磌�?1&t��16@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatka�9͢?!����69@)�<L��?1NH�ᙸ4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�,'��?!�?���K@)�$y��Ñ?1�_�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceL8��+�?!���
5 @)L8��+�?1���
5 @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?���?!N��v�/@)37߈�Y�?1v<�u�P@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����z?!>.;$$�@)����z?1>.;$$�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���!o��?!M�{9n�1@)nYk(�g?1cbD�.��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 65.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�24.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�eM�i�?I؀��9�V@Q�L2�� !@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	TS�u8�.@TS�u8�.@!TS�u8�.@      ��!       "	�$�� @�$�� @!�$�� @*      ��!       2	_�BF��?_�BF��?!_�BF��?:	�}���@�}���@!�}���@B      ��!       J	���~�?���~�?!���~�?R      ��!       Z	���~�?���~�?!���~�?b      ��!       JGPUY�eM�i�?b q؀��9�V@y�L2�� !@�"1
model/Conv1D_2/conv1dConv2D%��ɂ�?!%��ɂ�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputUX��8�?!=��]�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�z4���?!����G�?0"1
model/Conv1D_3/conv1dConv2D5�����?!!�֪�v�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�k|�?!"��!x��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputc"�Pd��?!nQ�����?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�%���2�?!���҉�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradWw���?!C��l���?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradWw���?!�Rmdm�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�����?!a������?Q      Y@Y��ί=�&@a.+Jx#V@q?FK�jbT@y�#*P���?"�
both�Your program is POTENTIALLY input-bound because 65.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�24.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�81.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 