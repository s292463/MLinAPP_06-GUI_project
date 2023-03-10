�	8�q��.@8�q��.@!8�q��.@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8�q��.@~7ݲC�?1T�D�[J'@A�^Ӄ�R�?I��"j�O�?rEagerKernelExecute 0*	���MbLe@2F
Iterator::Model�O �Ȓ�?!Xў���H@)�?�,խ?1c�l�2A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr��9��?!3q��6�;@)�)���?1d�F֜6@:Preprocessing2U
Iterator::Model::ParallelMapV2��B�ʠ�?!���,�.@)��B�ʠ�?1���,�.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceKXc'�?!�	��" @)KXc'�?1�	��" @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQj/�혚?!��d�|.@)��)x
�?1�&��=�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipڎ����?!�.aMEI@)c��*3��?1�W����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�zM
J�?!8#����@)�zM
J�?18#����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�N��C�?!&Vo�0@)$��Pe?1�y �o�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt�!�H8@Q�w���R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~7ݲC�?~7ݲC�?!~7ݲC�?      ��!       "	T�D�[J'@T�D�[J'@!T�D�[J'@*      ��!       2	�^Ӄ�R�?�^Ӄ�R�?!�^Ӄ�R�?:	��"j�O�?��"j�O�?!��"j�O�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qt�!�H8@y�w���R@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter`���,!�?!`���,!�?0"1
model/Conv1D_2/conv1dConv2Dt1���?!�Q��zw�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��R���?!�^��z�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter"Dƛ��?!
sPg4�?0"1
model/Conv1D_3/conv1dConv2D�GE��?!���Q��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�2yF숨?!�`mo��?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradX�G�ޢ?!��N�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�����?!d~x��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�v2ę�?!�����?"3
model/Conv1D_1/BiasAddBiasAdd���My�?!h�'�9��?Q      Y@Y.>9\&@a<�?�x4V@q�^�G|0@y��CxSU�?"�
both�Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�16.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 