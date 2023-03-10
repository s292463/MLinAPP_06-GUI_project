�	�ٕF@�ٕF@!�ٕF@	Ё�T-@Ё�T-@!Ё�T-@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�ٕF@����?1ܼqR��@A��-��?I��1@Y	�L�n�?rEagerKernelExecute 0*	X9�Ȥw@2U
Iterator::Model::ParallelMapV2�+����?!�p�9�N@)�+����?1�p�9�N@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenated"��<�?!Q u���+@)���J��?1��U9~)@:Preprocessing2F
Iterator::Model�/J�_��?!s�۾}R@)+3����?1XUTv�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8I�Ǵ6�?!4���	:@)��4Ԙ?19���N�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(z�c��?!��HGL@)XWj1x�?1�U�LD	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor<�ݭ,�y?!�r9섨�?)<�ݭ,�y?1�r9섨�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��O��?!|�~>�-@) �o_�i?1��`E��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensormU�Yf?!r���o�?)mU�Yf?1r���o�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\���4_?!T�a��?)\���4_?1T�a��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ё�T-@I����P@Q���>-�=@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����?����?!����?      ��!       "	ܼqR��@ܼqR��@!ܼqR��@*      ��!       2	��-��?��-��?!��-��?:	��1@��1@!��1@B      ��!       J		�L�n�?	�L�n�?!	�L�n�?R      ��!       Z		�L�n�?	�L�n�?!	�L�n�?b      ��!       JGPUYЁ�T-@b q����P@y���>-�=@�"1
model/Conv1D_2/conv1dConv2D���u�?!���u�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput=�Vx�-�?!*����?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterΔ9-h�?!6��SD��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~�.f�p�?!��m``y�?0"1
model/Conv1D_3/conv1dConv2D^����a�?!�j,\���?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput���z��?!r��	Ǐ�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad|�H�i=�?!�Eʨ�C�?"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam>j���>�?!��S(���?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~陘�ʗ?!6k�/T�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad>��{�?!�k�b���?Q      Y@Yp���*@a��Ǐ�U@qJ\n;<;<@y���З��?"�
both�Your program is POTENTIALLY input-bound because 22.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�44.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�28.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 