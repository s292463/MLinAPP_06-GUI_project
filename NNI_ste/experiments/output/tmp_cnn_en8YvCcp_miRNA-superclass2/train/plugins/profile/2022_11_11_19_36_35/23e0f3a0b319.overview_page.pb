� 	en��S@en��S@!en��S@	��ߙo@��ߙo@!��ߙo@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLen��S@X����?1k�*�@A��~��Γ?II�V��@Y,��̰�?rEagerKernelExecute 0*	rh��|��@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�K�;���?!_�A9�kH@)������?1�y!�NF@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map�����S�?!豼�=�C@)�`���?1@��2�A@:Preprocessing2F
Iterator::Model:�6U�Ȳ?!"��k� @)�p��|#�?1��KW3@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat��Ӝ�Ȥ?!F�/Ѫr@)�_��ME�?1߇o�7@:Preprocessing2U
Iterator::Model::ParallelMapV2DkE��ܖ?!;�� K@)DkE��ܖ?1;�� K@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate=��@fg�?!{Fژ1��?)�h�^�?1�<�;��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicer����?!5���6i�?)r����?15���6i�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!����?!��3��@)�:9Cqǋ?1:��_��?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch#�=��?!h<I��?)#�=��?1h<I��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�0� O�?!�EEz�J@)�LM�7��?1�t(Q�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorG�ŧ x?!�'�x�M�?)G�ŧ x?1�'�x�M�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�	L�ut?!/+ ����?)�	L�ut?1/+ ����?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�v��-u�?!��&b47�?)���V%a?1��o�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorRC��P?!� ԵX�?)RC��P?1� ԵX�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�40.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��ߙo@I��$�`M@Q!<��hC@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	X����?X����?!X����?      ��!       "	k�*�@k�*�@!k�*�@*      ��!       2	��~��Γ?��~��Γ?!��~��Γ?:	I�V��@I�V��@!I�V��@B      ��!       J	,��̰�?,��̰�?!,��̰�?R      ��!       Z	,��̰�?,��̰�?!,��̰�?b      ��!       JGPUY��ߙo@b q��$�`M@y!<��hC@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterÏձ^�?!Ïձ^�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradR�(�ch�?!��t>Ȋ�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�8�q�?!&Nt	'�?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�.�!�?!�Y�GM��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputa�j!�?!
rU���?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��̙Y3�?!{Id���?"1
model/Conv1D_1/conv1dConv2D O��f�?!|9�ce��?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose[�W�^�?!��~|F��?"1
model/Conv1D_2/conv1dConv2D�Y:8�?!�Ed�[�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose[3П���?!%Ia���?Q      Y@Y�0�03@a��<��<T@q/ύt3@y"�u8B��?"�
both�Your program is POTENTIALLY input-bound because 17.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�40.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�19.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 