�	~����@~����@!~����@		4��H@	4��H@!	4��H@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL~����@vT5A�}�?1>���d@A���0�?I�y��@YE�N����?rEagerKernelExecute 0*	��x�&�c@2F
Iterator::Model�vhX���?!�;xt%G@)��S �g�?1�O�m-)>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateq9^��I�?!.hjr�@?@)�^a����?10
lf==@:Preprocessing2U
Iterator::Model::ParallelMapV2��`�?!�䂻!0@)��`�?1�䂻!0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY�.���?!�ć��J@),G�@��?1f��JZM"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�#Di�?!��Nw��&@)D���XP�?13��h�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�?mT�y?!�m��@)�?mT�y?1�m��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�D2�ت?!Ĩ:�@@)�g^��h?1����{��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��N]?!݃}���?)��N]?1݃}���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��|	\?!\h�^Z�?)��|	\?1\h�^Z�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9	4��H@IFl3S�O@Qz7��+�@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	vT5A�}�?vT5A�}�?!vT5A�}�?      ��!       "	>���d@>���d@!>���d@*      ��!       2	���0�?���0�?!���0�?:	�y��@�y��@!�y��@B      ��!       J	E�N����?E�N����?!E�N����?R      ��!       Z	E�N����?E�N����?!E�N����?b      ��!       JGPUY	4��H@b qFl3S�O@yz7��+�@@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterW	``9�?!W	``9�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�f{�ب?!jީ���?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad����-��?!^ҁ�}�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeX��?��?!����%�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposee�Ƙ|�?!�B�����?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposej����ǡ?!�V���?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��Ou�Y�?!R�\:�?0"3
model/Conv1D_1/BiasAddBiasAddm�s��Š?!��7��R�?"-
model/Conv1D_1/ReluRelu���3�?!����5Y�?"1
model/Conv1D_2/conv1dConv2D�eg��?!���f\�?Q      Y@Y�%~F�+@a�\;0׎U@q6W򤜦E@ye�Ƙ|�?"�
both�Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�43.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 