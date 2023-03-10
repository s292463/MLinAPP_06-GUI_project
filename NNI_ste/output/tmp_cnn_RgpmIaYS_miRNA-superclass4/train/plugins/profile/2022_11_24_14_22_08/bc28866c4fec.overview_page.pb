�	������*@������*@!������*@	���b��?���b��?!���b��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL������*@?�a�'�?1���X$@A����t�?I$d �.��?Y���Fu:�?rEagerKernelExecute 0*	㥛� �c@2F
Iterator::Model4h��b�?!��c,�TJ@):tzލ�?1�x���A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����,A�?!_=���f;@)ۧ�1��?17[�5@:Preprocessing2U
Iterator::Model::ParallelMapV2Z���f��?!Q��YB�0@)Z���f��?1Q��YB�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice^���?!��F��@)^���?1��F��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��X����?!����@)��X����?1����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI�\߇��?!���A)@)~�[�~l�?1N��P3�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipsJ_9�?!(+��	�G@)r�Z|
��?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB]¡�?!Z0k�-@)R~R���h?1�c򉒶�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���b��?Iě=���6@Q��$���R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	?�a�'�??�a�'�?!?�a�'�?      ��!       "	���X$@���X$@!���X$@*      ��!       2	����t�?����t�?!����t�?:	$d �.��?$d �.��?!$d �.��?B      ��!       J	���Fu:�?���Fu:�?!���Fu:�?R      ��!       Z	���Fu:�?���Fu:�?!���Fu:�?b      ��!       JGPUY���b��?b qě=���6@y��$���R@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter{�U(D��?!{�U(D��?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad,��Nݥ?!��묻?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrads62Q���?!%�� ���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��N~V�?!�5Q�N�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�Ƙ���?!ւ�ű�?0"1
model/Conv1D_4/conv1dConv2D{h�O��?!9�wz,��?"1
model/Conv1D_2/conv1dConv2DŶAd�?!�ľ��?"3
model/Conv1D_1/BiasAddBiasAdd^ecAxʟ?!KE�B��?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose��E8c�?!3ǽ
�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposemT�NY�?!R��R �?Q      Y@Y������'@aV@q�cxa��6@y��]���?"�
both�Your program is POTENTIALLY input-bound because 9.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�22.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 