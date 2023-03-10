�	��J�@��J�@!��J�@	�� �@�� �@!�� �@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��J�@��6���?1(v�U� @A���L��?Ij��j	@Y���z�2�?rEagerKernelExecute 0*	����.�@2U
Iterator::Model::ParallelMapV2&���J�?!�K��J@)&���J�?1�K��J@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS]���?!p�6&�C@)�R\U�]�?1�� v�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�J�h�?!��CQ�. @)�4�($�?1{�"#T�@:Preprocessing2F
Iterator::Model3��yS�?!�4���N@)g��F�?1�F�m��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceu;�ʃ��?!�X���@)u;�ʃ��?1�X���@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'��rJ@�?!3����I@)�v���?1��Fq<�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�X���?!c���	��?)�X���?1c���	��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��F;n��?!�I���-@)@L<�k?1赫YF�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�42.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�� �@I~L�d^5P@Q_�~g��;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��6���?��6���?!��6���?      ��!       "	(v�U� @(v�U� @!(v�U� @*      ��!       2	���L��?���L��?!���L��?:	j��j	@j��j	@!j��j	@B      ��!       J	���z�2�?���z�2�?!���z�2�?R      ��!       Z	���z�2�?���z�2�?!���z�2�?b      ��!       JGPUY�� �@b q~L�d^5P@y_�~g��;@�"1
model/Conv1D_3/conv1dConv2D�rm?xo�?!�rm?xo�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterv5�e$�?!mtQ���?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�ŕ.�?!���qU�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterK���2�?!���
��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�֎�z�?!��CM@�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput���x��?!܈<R�}�?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad�ʸS0t�?!7��\��?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsU���?!"#e�$��?"1
model/Conv1D_2/conv1dConv2Dq�v��?!���9�}�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�y1�$��?!8�/��U�?0Q      Y@Y�;�;-@a��؉�XU@q�ip�:@y�4�zk�?"�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�42.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t21.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�26.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 