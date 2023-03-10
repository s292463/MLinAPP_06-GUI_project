�	{�\�&q@{�\�&q@!{�\�&q@	���C�@���C�@!���C�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL{�\�&q@���b��?1�ި�/@A�� �K�?IyX�5�;	@YXXp?���?rEagerKernelExecute 0*	��n��e@2F
Iterator::Model��Bs�F�?!7"j}d�D@)��:��?1x�G�M$=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�r��ǩ?!�)�h@=@)�Ӻj��?1�r�Ԭ8@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM
J�ʹ?!�ݕ��CM@)���·g�?14X����,@:Preprocessing2U
Iterator::Model::ParallelMapV2�.n���?!�3���(@)�.n���?1�3���(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices�蜟�?!�#�(<@)s�蜟�?1�#�(<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��\QJ�?!���D��(@)Y���-�?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb��BW"�?!��QN@)b��BW"�?1��QN@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!���3�?!Ep��-@)��D-ͭp?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���C�@Il���*P@Q��EP?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���b��?���b��?!���b��?      ��!       "	�ި�/@�ި�/@!�ި�/@*      ��!       2	�� �K�?�� �K�?!�� �K�?:	yX�5�;	@yX�5�;	@!yX�5�;	@B      ��!       J	XXp?���?XXp?���?!XXp?���?R      ��!       Z	XXp?���?XXp?���?!XXp?���?b      ��!       JGPUY���C�@b ql���*P@y��EP?@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterKyV��O�?!KyV��O�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradCZ%�?!Yx�vb�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�&����?!N�A����?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����¤?!H�m2\��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�#��'��?!B�b/&	�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputh�8�l�?!�v8�@��?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradڊ�:�X�?!)�o\��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�+1?A�?!�-YQ���?"1
model/Conv1D_2/conv1dConv2De��H�?�?!#6��o�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputA��ኜ?!���08�?0Q      Y@Y��u@7�)@a%D�9�U@q���B@y��1�J��?"�
both�Your program is POTENTIALLY input-bound because 18.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�46.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�37.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 