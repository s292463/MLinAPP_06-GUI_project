�	���C�x@���C�x@!���C�x@	Y��S)֩?Y��S)֩?!Y��S)֩?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���C�x@�A'�:�?1���Kv@A�x��?IcE�a�D@Y�A�f��?rEagerKernelExecute 0*	V-��+c@2F
Iterator::Model�jf-��?!q��b�K@)�3��k�?1�v\�B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateW�[Ɏ�?!I�22Xt;@) Tq��?1�w	h9@:Preprocessing2U
Iterator::Model::ParallelMapV2RH2�w��?!�*d��2@)RH2�w��?1�*d��2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�z��9y�?!��f;�@&@)���y�?1vK��>P@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�1 Ǟ�?!��W�pF@)�9��!|?1�1���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor$�����w?!��=�b@)$�����w?1��=�b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��`U���?!D���eF=@)��?�f?1����� �?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��� �Y?!VP ,�{�?)��� �Y?1VP ,�{�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceaU��N�Y?!I� vI�?)aU��N�Y?1I� vI�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y��S)֩?I�0=�S�%@Qy�m�ZLV@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�A'�:�?�A'�:�?!�A'�:�?      ��!       "	���Kv@���Kv@!���Kv@*      ��!       2	�x��?�x��?!�x��?:	cE�a�D@cE�a�D@!cE�a�D@B      ��!       J	�A�f��?�A�f��?!�A�f��?R      ��!       Z	�A�f��?�A�f��?!�A�f��?b      ��!       JGPUYY��S)֩?b q�0=�S�%@yy�m�ZLV@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�6��p��?!�6��p��?0"1
model/Conv1D_2/conv1dConv2D6rA��?!p��B��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputE��0˚�?!��(�=�?0"1
model/Conv1D_3/conv1dConv2Dv����?!V�x1;�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�⑴e�?!��9��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�8�^��?!�B��<��?0"1
model/Conv1D_4/conv1dConv2De���	��?!	ǝ@�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�8Zೋ?!���ܮ�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�9-�Y�?!�^gC�?0"1
model/Conv1D_1/conv1dConv2D���!72�?!`n�C��?Q      Y@Y�{�1m@aD�,��W@q.,�	Z@@y ��wH?"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�32.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 