�	ꗈ��O@ꗈ��O@!ꗈ��O@	�Zl6��@�Zl6��@!�Zl6��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLꗈ��O@�G��[�?1P8��L��?A�����?I9���I@Y��0|D�?rEagerKernelExecute 0*	/�$��@2U
Iterator::Model::ParallelMapV2���4}�?!����?_S@)���4}�?1����?_S@:Preprocessing2F
Iterator::Modelc('�UH�?!Η(��dU@)��K����?1Z�_�m* @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS���"��?!}r��j@)�]��-ɡ?1�4,�-^@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$���9"�?!�	8��s@)6��đ?1 u�RX@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��-���?!��d�T�@)��-���?1��d�T�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip������?!�A����,@)�R��?1�����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorPn����~?!+�3�?)Pn����~?1+�3�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�J��?![���k@)g��j+�g?1����[{�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�50.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t23.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�Zl6��@IU]�'wbR@Q�s�$3@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�G��[�?�G��[�?!�G��[�?      ��!       "	P8��L��?P8��L��?!P8��L��?*      ��!       2	�����?�����?!�����?:	9���I@9���I@!9���I@B      ��!       J	��0|D�?��0|D�?!��0|D�?R      ��!       Z	��0|D�?��0|D�?!��0|D�?b      ��!       JGPUY�Zl6��@b qU]�'wbR@y�s�$3@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�ޏ�[}�?!�ޏ�[}�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradT�����?!�� �?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�����?!�e ���?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���ч��?!��u(g�?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose������?!�ҥ�MN�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose �wp�Ѣ?!:�apd�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposeb�~ˢ?!~�81�Z�?"3
model/Conv1D_1/BiasAddBiasAdd��f��?!M���?"-
model/Conv1D_1/ReluRelue��[��?!�x��?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose���587�?!t9�~���?Q      Y@Yyxxxxx*@a�����U@q��*~~94@y��J�m�?"�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�50.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t23.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�20.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 