�	X�<���0@X�<���0@!X�<���0@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCX�<���0@�9�m�@1Sx�캷@A�{b�*ߓ?I�8'0E!@rEagerKernelExecute 0*	�Q���t@2Z
#Iterator::Model::ParallelMapV2::ZipD��k��?!v)�,�S@)r3܀��?1}p*�/L@:Preprocessing2F
Iterator::Modelb�G�?!)ZM_�7@)�*�MF��?1s)I.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�n��;��?!vaVZ4�(@),g*�?1�%�P�R$@:Preprocessing2U
Iterator::Model::ParallelMapV2�2Q���?!>��~�!@)�2Q���?1>��~�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���P���?!lNX�@)���P���?1lNX�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?9
3�?!���ƫG@)��g�ej�?1]�X?��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�;��)t~?!~�|%�@)�;��)t~?1~�|%�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�W�\�?!�8��H@)�fF?Ni?1wf5����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�51.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�)�G�S@Q�Y���7@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�9�m�@�9�m�@!�9�m�@      ��!       "	Sx�캷@Sx�캷@!Sx�캷@*      ��!       2	�{b�*ߓ?�{b�*ߓ?!�{b�*ߓ?:	�8'0E!@�8'0E!@!�8'0E!@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�)�G�S@y�Y���7@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<Y'��{�?!<Y'��{�?0"1
model/Conv1D_2/conv1dConv2D��k@Ÿ�?!]A?X�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��%�p��?!wI3��"�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�?���?!�D�x���?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradr8
G�Ǥ?!���!xs�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter%����?!��z�Y��?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposey���A��?!��/���?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose!�do�?!��ϵ���?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposebR�gT�?!��G&���?"3
model/Conv1D_1/BiasAddBiasAdd�J~�T��?!y��r���?Q      Y@Y��u@7�)@a%D�9�U@q��Rn�R@y`�9���?"�
both�Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�51.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�74.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 