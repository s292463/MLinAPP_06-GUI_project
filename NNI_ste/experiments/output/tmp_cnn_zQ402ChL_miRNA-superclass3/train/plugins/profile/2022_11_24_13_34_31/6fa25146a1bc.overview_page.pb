�	I�2��@I�2��@!I�2��@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCI�2��@�WWj��?1�{c ��?A�EИ�?INA~6rm@rEagerKernelExecute 0*	{�G�zc@2F
Iterator::Modeli����ѳ?!�^Դ�H@)p�'v��?1�<�*��B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO�����?!�s<k��8@)��8~�?1-�aʩ�4@:Preprocessing2U
Iterator::Model::ParallelMapV2�W�2�?!�� )~P)@)�W�2�?1�� )~P)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9�j�3�?!a�+K�(I@)bjK���?1<k���A%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�T�=ϟ�?!te��Z@)�T�=ϟ�?1te��Z@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatesh��|?�?!�e�?�*@)�{b�*߃?1�f;��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���0y?!�k�h@)���0y?1�k�h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�5���?!�2�͟0.@)�k���f?1h��{�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�55.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�OJ��S@QS���V�4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�WWj��?�WWj��?!�WWj��?      ��!       "	�{c ��?�{c ��?!�{c ��?*      ��!       2	�EИ�?�EИ�?!�EИ�?:	NA~6rm@NA~6rm@!NA~6rm@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�OJ��S@yS���V�4@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��,,p¾?!��,,p¾?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri|&���?!��)%�?0"1
model/Conv1D_2/conv1dConv2D�^40b�?!L���~�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterO�� /�?!6D3���?0"1
model/Conv1D_3/conv1dConv2DjE��oF�?!���-�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad����L��?!�~ۭ�A�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad����L��?!Q�D�U�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�~�8�ߟ?!�X_��S�?0"1
model/Conv1D_1/conv1dConv2Dd�*�7[�?!;�vA�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputpML'�?!��v�C��?0Q      Y@YD+l$Z)@a�z2~��U@q)�l��K@y��X�:��?"�
both�Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�55.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�56.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 