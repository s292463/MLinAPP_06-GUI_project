�	��~��&#@��~��&#@!��~��&#@	@E/9@@E/9@!@E/9@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��~��&#@�>����?1��o��6@AJ~įXÕ?I��:��?Y>��I���?rEagerKernelExecute 0*	�n��Bc@2F
Iterator::Model.��ϱ?!y�拓F@)��mRѨ?1��
�/u?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����zi�?!Xh�BF�@@)%vmo�$�?1�ˤ�U=@:Preprocessing2U
Iterator::Model::ParallelMapV2Q�?Û�?!�6n�c+@)Q�?Û�?1�6n�c+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL������?!P�L	�*@).��?1�1��P@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!�mtlK@)��s�v��?1\�n 
J@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicej���v��?!�dh]K�@)j���v��?1�dh]K�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?�{�&z?!?�d���@)?�{�&z?1?�d���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����X�?!�3Z��-@)�0e��f?1�4�:��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�17.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9@E/9@IǞBq
d@@Q�x����O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�>����?�>����?!�>����?      ��!       "	��o��6@��o��6@!��o��6@*      ��!       2	J~įXÕ?J~įXÕ?!J~įXÕ?:	��:��?��:��?!��:��?B      ��!       J	>��I���?>��I���?!>��I���?R      ��!       Z	>��I���?>��I���?!>��I���?b      ��!       JGPUY@E/9@b qǞBq
d@@y�x����O@�"1
model/Conv1D_3/conv1dConv2D��\�g�?!��\�g�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput���:��?!6�#�˿?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter"VW�?!�}����?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterx�\ϩ�?!�f{"&�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad �g��X�?!=2�."��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�����E�?!0Dv��&�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad}$��̠?!�~w@�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��\>DZ�?!=���K�?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����rӟ?!��/�H�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose8eBߝ?!j8Q�&�?Q      Y@Y�
*T�(@a�^�z��U@qF3C��E2@y�L��˦?"�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�17.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�18.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 