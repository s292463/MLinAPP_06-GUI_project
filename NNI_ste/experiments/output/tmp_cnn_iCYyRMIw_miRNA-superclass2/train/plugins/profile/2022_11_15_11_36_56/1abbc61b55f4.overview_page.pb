�	��Z&��@��Z&��@!��Z&��@	��@��@!��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��Z&��@Hm��~G�?1�r߉�?A��bc^�?I����@Y�v�$�?rEagerKernelExecute 0*	V-��d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[Υ���?![p=��e@@) c�ZB>�?1h��B�x<@:Preprocessing2F
Iterator::ModelZ*oG8-�?!dݫ:�G@)� �	��?1~�#�3�;@:Preprocessing2U
Iterator::Model::ParallelMapV2�S �gР?!I�3+A�3@)�S �gР?1I�3+A�3@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�C�bԵ�?!9�����@)�C�bԵ�?19�����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���C�r�?!�E��0)@)}�%�/�?1��Zb�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����e�?!�"T��MJ@)���.\�?1^n���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�Y��Bs}?!9e\62K@)�Y��Bs}?19e\62K@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapscz��?!����rA,@)�nf���d?1��
��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�53.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��@I�����XR@QH�e�E8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Hm��~G�?Hm��~G�?!Hm��~G�?      ��!       "	�r߉�?�r߉�?!�r߉�?*      ��!       2	��bc^�?��bc^�?!��bc^�?:	����@����@!����@B      ��!       J	�v�$�?�v�$�?!�v�$�?R      ��!       Z	�v�$�?�v�$�?!�v�$�?b      ��!       JGPUY��@b q�����XR@yH�e�E8@�"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter.5m%��?!.5m%��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�V:kE�?!�`�"ȕ�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��RHo«?!�\�1C�?0"1
model/Conv1D_3/conv1dConv2D9�O��0�?!���P��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad.�8]Ч?!-9S��?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad������?!�K���?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput9��� �?!����I��?0"1
model/Conv1D_2/conv1dConv2D��&ۿ�?!}dR�<�?"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�b!�:��?!U=���j�?"}
^gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposej�u�-�?!��u��3�?Q      Y@Y@n]�G*@a8R4��U@qE&��>@yf��}L1�?"�
both�Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�30.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 